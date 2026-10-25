# The draft head's context rows: selected rows of a row-split head by id, and their merges into a
# proposal argmax or top sixteen (include/ninfer/ops/linear_rows.h).
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/linear_rows.cu")
