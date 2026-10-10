# offloaded_sparse_moe over GGML expert records (include/ninfer/ops/offloaded_sparse_moe.h).
#
# Host routes: the CPU expert engine (scalar and AVX2), its worker team and the miss service. Their
# canonical arithmetic (src/ops/common/canonical_ggml.h) must compile without floating-point
# contraction or fast math on every compiler, or the CPU and GPU bits would differ.
set(ninfer_offloaded_moe_host_sources
  "${CMAKE_CURRENT_LIST_DIR}/cpu/expert_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/expert_avx2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/expert_team.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/miss_service.cpp")
target_sources(ninfer_ops PRIVATE ${ninfer_offloaded_moe_host_sources})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(${ninfer_offloaded_moe_host_sources} PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
elseif(MSVC)
  set_source_files_properties(${ninfer_offloaded_moe_host_sources} PROPERTIES COMPILE_OPTIONS "/fp:precise")
endif()

# GPU: routing, dispatch, staging, the narrow expert route, the CPU channel and the combine.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cuda/moe_layer.cu")
