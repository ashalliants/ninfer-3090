# Standalone CUDA enumeration probe. It links nothing from NInfer on purpose: the whole point is
# to answer "which CUDA indices are the two cards, and can they reach each other" before the model
# loader is involved, on a machine where the answer is not obvious.
#
# A display GPU commonly takes cuda=0 even when nvidia-smi lists it elsewhere, so the indices for
# --devices must be read from a CUDA-level enumeration rather than assumed.
add_executable(ninfer-multi-gpu-probe multi_gpu_probe.cu)
set_target_properties(ninfer-multi-gpu-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
