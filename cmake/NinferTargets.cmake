# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${NINFER_CUDA_RDC}
    CUDA_RESOLVE_DEVICE_SYMBOLS ${NINFER_CUDA_RDC})
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

function(ninfer_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# NINFER_HOST_ONLY (no CUDA compiler): the target's .cu sources compile as C++ against the CUDA
# headers, except those named after KERNELS, which define kernels and are left out of the target.
function(ninfer_host_only_sources target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "KERNELS")
  get_target_property(sources ${target} SOURCES)
  set(kept "")
  foreach(source IN LISTS sources)
    if(source IN_LIST arg_KERNELS)
      continue()
    endif()
    if(source MATCHES "\\.cu$")
      set_source_files_properties(${source} TARGET_DIRECTORY ${target} PROPERTIES LANGUAGE CXX)
    endif()
    list(APPEND kept ${source})
  endforeach()
  set_property(TARGET ${target} PROPERTY SOURCES ${kept})
endfunction()
