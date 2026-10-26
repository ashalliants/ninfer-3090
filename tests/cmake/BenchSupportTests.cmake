# NEEDS_SOURCE_DIR: the community preset's checks read the committed corpus and chat prompts.
ninfer_add_test(ninfer_bench_support_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_bench_support.cpp"
          ${PROJECT_SOURCE_DIR}/bench/inference/ninfer_bench_support.cpp
  LIBRARIES ninfer_engine ninfer::json)

target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)

ninfer_add_op_test(ninfer_bench_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_bench_fixtures.cu"
  LIBRARIES ninfer_ops)
target_include_directories(ninfer_bench_fixtures_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

add_executable(ninfer_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# The prefill stall probe's self-test (tools/bench/run_prefill_stall.py), against a fake server that
# stalls its streams while a long request is "prefilling". Pure Python, no GPU.
add_test(NAME ninfer_prefill_stall_probe_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_prefill_stall.py)

# The prefill time model (tools/bench/prefill_time_model.py) and the depth sweep that checks it against
# a card (tools/bench/run_prefill_depth_sweep.py). Pure Python, no GPU: the model is checked against the
# measurements docs/performance.md records and against the C++ PrefillPacing::step_cap; the sweep against
# a fake ninfer_bench.
add_test(NAME ninfer_prefill_time_model_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_prefill_time_model.py)
add_test(NAME ninfer_prefill_depth_sweep_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_prefill_depth_sweep.py)

# The Host arena replay tool's self-test (tools/bench/replay_host_arena.py). Pure Python, no GPU: the
# replay of a known allocator run must reproduce it exactly, or the tool's predictions mean nothing.
add_test(NAME ninfer_host_arena_replay_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_replay_host_arena.py)

# The community rows checker's self-test (tools/bench/check_bench_rows.py): a good file passes, each
# malformed row fails, --fix rewrites the house format, and the committed bench/community/rows.json
# passes. Pure Python, no GPU.
add_test(NAME ninfer_bench_rows_check_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/check_bench_rows.py --self-test)
