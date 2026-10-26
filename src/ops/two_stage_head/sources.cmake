# The two-stage output head (--lm-head-two-stage, include/ninfer/ops/two_stage_head.h): the 2-bit
# copy scored on the int8 tensor cores, the column thresholds, and the rows rescored from the Q8 head.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/two_stage_head.cu")
