# Query-sparse attention: index-query preparation, pooled index keys, block selection and attention
# over the selected blocks (include/ninfer/ops/qsa.h).
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/qsa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prep.cu"
  "${CMAKE_CURRENT_LIST_DIR}/select_tiled.cu"
  "${CMAKE_CURRENT_LIST_DIR}/select_sliced.cu"
  "${CMAKE_CURRENT_LIST_DIR}/attention_split.cu"
  "${CMAKE_CURRENT_LIST_DIR}/attention_wide.cu"
)