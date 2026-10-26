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

# The lossy prefill quality/speed sweep (tools/bench/run_lossy_prefill_sweep.py): its summariser against
# fixture results (the acceptance numbers' edges), grid expansion and command construction. No server, no GPU.
add_test(NAME ninfer_lossy_prefill_sweep_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_lossy_prefill_sweep.py)

# The Host arena replay tool's self-test (tools/bench/replay_host_arena.py). Pure Python, no GPU: the
# replay of a known allocator run must reproduce it exactly, or the tool's predictions mean nothing.
add_test(NAME ninfer_host_arena_replay_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_replay_host_arena.py)

# The first-principles performance model (tools/perf_model.py): checked against the byte counts of
# decode_byte_accounting.py, the RTX 3090 rows of docs/performance.md and the Engine's fitted prefill
# cost table. Pure Python, no GPU.
add_test(NAME ninfer_perf_model_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/test_perf_model.py)

# The tools built on it: the nsys kernel-time join (tools/kernel_bandwidth.py, against synthetic CSVs in
# nsys's layout) and the Flash-Next component model (tools/flash_next_perf_model.py, against the
# measurements in docs/maintainer/flash-next-plan.md and the lanes spike). Pure Python, no GPU.
add_test(NAME ninfer_kernel_bandwidth_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/test_kernel_bandwidth.py)
add_test(NAME ninfer_flash_next_perf_model_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/test_flash_next_perf_model.py)

# The rental script's power-limit guard (scripts/vast/power_guard.py): a Vast host can cap the card well
# below its default and the offer listing does not say so. Pure Python, no GPU and no Vast account.
add_test(NAME ninfer_vast_power_guard_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/scripts/vast/test_power_guard.py)

# The rental loop itself (scripts/vast/vast-up.sh) against a fake vastai and ssh: a rejected host is
# destroyed and the next tried, a run with no good host leaves nothing billing, and a failed destroy keeps
# the instance tracked. It runs the bash script, so it is registered where bash is the shell (not Windows).
if(UNIX)
  add_test(NAME ninfer_vast_up_test
    COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/scripts/vast/test_vast_up.py)
endif()

# The community rows checker's self-test (tools/bench/check_bench_rows.py): a good file passes, each
# malformed row fails, --fix rewrites the house format, and the committed bench/community/rows.json
# passes. Pure Python, no GPU.
add_test(NAME ninfer_bench_rows_check_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/check_bench_rows.py --self-test)

# The randomised soak harness's self-test (tools/bench/serve_soak.py), against a fake server that stalls,
# answers 400/500, grows its resident memory and goes away. Pure Python, no GPU, about 12 s.
add_test(NAME ninfer_serve_soak_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_serve_soak.py)

# The stderr ratchet (tools/check_stderr_logging.py): its self-test on temporary trees (a rise or a new
# file fails, a drop passes and --update lowers it, comments and strings are not counted), and the real
# check of src/ against tools/stderr_logging.allow, which is what makes CI refuse a new direct write.
# Pure Python, no GPU.
add_test(NAME ninfer_stderr_logging_check_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/test_check_stderr_logging.py)
add_test(NAME ninfer_stderr_logging_ratchet_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/check_stderr_logging.py
          --root ${PROJECT_SOURCE_DIR})

# The greedy identity tool's self-test (tools/bench/greedy_identity.py), against fake CLIs that agree,
# diverge at a chosen token, stop early, crash or print no ids. Pure Python, no GPU.
add_test(NAME ninfer_greedy_identity_test
  COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tools/bench/test_greedy_identity.py)
