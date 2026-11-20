# The pooling step of DeepSeek-V4's KV compressor.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/softmax_pool.cu")
