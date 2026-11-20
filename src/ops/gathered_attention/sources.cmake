# DeepSeek-V4's attention over gathered KV rows with a softmax sink (one shared KV head).
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/gathered_sink_attention.cu")
