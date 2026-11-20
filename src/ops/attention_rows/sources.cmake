# Row bookkeeping of DeepSeek-V4's attention layer: reachability, row lists and candidate blocks.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/attention_rows.cu")
