add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  companion.cpp
  materializer.cpp
  transcode.cpp
  permute_row_split.cu
  views.cpp
)
if(NINFER_HOST_ONLY)
  # See src/core/CMakeLists.txt.
  ninfer_host_only_sources(ninfer_artifact KERNELS permute_row_split.cu)
endif()
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)

# The 2-bit head copy (companion.cpp) runs the canonical codec of src/ops/two_stage_head; it must
# compile without floating-point contraction or fast math, as the GPU test's host oracle does.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(companion.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
elseif(MSVC)
  set_source_files_properties(companion.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
endif()
