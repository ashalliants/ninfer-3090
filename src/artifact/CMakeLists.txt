add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
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
