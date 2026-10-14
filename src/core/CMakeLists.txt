add_library(ninfer_core STATIC
  arena.cu
  device.cu
  stage_link.cu
  host_gather.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  evictable_kv_pool.cu
  evictable_weight_pool.cu
  vmm_arena.cu
  vmm_region.cu
  device_snapshot.cu
  pinned_upload.cu
  copy_batch.cpp
  cgroup_memory.cpp
  host_memory.cpp
  link_probe.cpp
  power_throttling.cpp
  vram_budget.cpp
)
if(WIN32)
  target_sources(ninfer_core PRIVATE read_only_file_win32.cpp scratch_file_win32.cpp)
  # vram_budget.cpp reads the OS video-memory budget through DXGI.
  target_link_libraries(ninfer_core PRIVATE dxgi)
else()
  target_sources(ninfer_core PRIVATE read_only_file_posix.cpp scratch_file_posix.cpp)
  # vram_budget.cpp loads libnvidia-ml.so.1 via dlopen/dlsym/dlclose.
  target_link_libraries(ninfer_core PRIVATE ${CMAKE_DL_LIBS})
endif()
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ${NINFER_CUDART_TARGET} CUDA::nvtx3 Threads::Threads CUDA::cuda_driver)
