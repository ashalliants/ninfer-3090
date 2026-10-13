# Qwen3.8-Flash-Next (qwen4exp). Host only: the n-gram hash against its fixture, and the n-gram
# volume reader on a synthetic volume. GPU, real artifact (skips without
# NINFER_TEST_QWEN4EXP_ARTIFACT): the whole forward pass, teacher-forced scoring and block taps.
ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): verification reproduces one-column
# decode bit for bit and its commit leaves the decode's state; the drafter's fusion, MoE and head
# against FP64 oracles on its tapped inputs.
ninfer_add_test(ninfer_qwen4_exp_mtp_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
set_tests_properties(ninfer_qwen4_exp_mtp_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Host only: a lane's three-axis RoPE positions (media prompts, later tokens at index + delta,
# pooled-block starts, call and MTP sub-chunk staging) and a call's visual columns.
ninfer_add_test(ninfer_qwen4_exp_rope_positions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rope_positions.cpp"
  LIBRARIES ninfer_model_runtime ninfer_core)

# GPU, real artifact with the vision tower (skips otherwise): the encode the Program runs (the tower
# streamed from the pinned block through a window) against an independent FP64 oracle on three
# images, with two negative controls the criterion must reject.
ninfer_add_test(ninfer_qwen4_exp_vision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
set_tests_properties(ninfer_qwen4_exp_vision_real_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the vision tower (skips otherwise): images through the public Engine
# (placement of the image tokens and their M-RoPE positions, answers on the example images, MTP
# exactness with images in context); --kv-cycle and --bench modes.
ninfer_add_test(ninfer_qwen4_exp_vision_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_engine_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen4_exp_vision_engine_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Host only: the expert cache (LFRU against the replay tool's decisions, frames, the controller and
# the warm-start state file) and the startup memory plans (RAM ledger, VRAM sizing, control law).
ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)

# Host only: the context cache's Host tier (slabs, checkpoints sharing KV pages, release quotes and
# sets, lineage recovery loss), the RAM ledger's sizing of it beside the experts, and the context
# store's checkpoint images (round trip, refusals, no-room and failing-tier misses).
ninfer_add_test(ninfer_qwen4_exp_context_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_cache.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

# GPU, no artifact: the context cache's NVMe tier on a scratch file (RAM frames, write-through and
# eviction, direct captures, host I/O, a failing disk, no frames); --bench DIR [MiB] measures it.
ninfer_add_test(ninfer_qwen4_exp_nvme_tier_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvme_tier.cpp"
  LIBRARIES ninfer_model_runtime ninfer_core)
set_tests_properties(ninfer_qwen4_exp_nvme_tier_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

# GPU, real artifact (skips without NINFER_TEST_QWEN4EXP_ARTIFACT): the M1 gates through the public
# Engine (placement invariance across cold, warm, warm-started and small expert caches; the RAM
# ledger's refusal; the elastic KV pool's grow/shrink cycle within the VRAM budget) and, with
# --bench, the decode measurements.
ninfer_add_test(ninfer_qwen4_exp_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen4_exp_engine_real_test PROPERTIES SKIP_RETURN_CODE 77)
