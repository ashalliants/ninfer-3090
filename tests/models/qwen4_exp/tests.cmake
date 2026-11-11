# Qwen3.8-Flash-Next (qwen4exp). Host only: the n-gram hash against its fixture, and the n-gram
# volume reader on a synthetic volume. GPU, real artifact (skips without
# NINFER_TEST_QWEN4EXP_ARTIFACT): the whole forward pass, teacher-forced scoring, block taps, the
# decode-path KLD gate (--kld-base, scored by ninfer-perplexity's evaluation code) and the bitwise
# identities of the shared expert on its side stream (--shared-expert-stream) and of the GDN's tiled
# head map (--gdn-head-map: prefill, a verification round with its fold, and decode logits).
ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

# Host only, header only: the draft-length policies (the EMA rule the Program uses and Basalt's
# joint-acceptance rule with its adaptive threshold and the window --mtp-joint-acceptance verifies).
ninfer_add_test(ninfer_qwen4_exp_draft_policy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_draft_policy.cpp"
  LIBRARIES ninfer_core)

# Host only, header only: the CPU miss service's error word (--cpu-verify-misses,
# --prefill-cpu-share): a code a thrown call left behind does not fail the next healthy call.
ninfer_add_test(ninfer_qwen4_exp_cpu_error_word_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_cpu_error_word.cpp"
  LIBRARIES ninfer_core)

# Host only, header only: a constrained MTP verification's window (--mtp-constrained): the
# positions before the first dead-end grammar mask.
ninfer_add_test(ninfer_qwen4_exp_constrained_window_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_constrained_window.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_perplexity_evaluation ninfer_core)
set_tests_properties(ninfer_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): verification reproduces one-column
# decode bit for bit and its commit leaves the decode's state; the drafter's fusion, MoE and head
# against FP64 oracles on its tapped inputs; the shared expert stream leaves the prompt's logits and
# the drafter's step bitwise unchanged.
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

# Host only: partial RAM residency's bookkeeping (--partial-ram-residency): a layer's RAM slots, the
# swap plan's write-backs and admissions, and random evict/admit rounds in which every expert keeps a
# valid copy and no frame is reused under a write-back.
ninfer_add_test(ninfer_qwen4_exp_ram_slots_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ram_slots.cpp"
  LIBRARIES ninfer_model_runtime ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)

# Host only: the Program's planned device terms computed from the options and the configuration
# and their startup checks (the shared expert's side-stream workspace against the Linear Op's own
# capacity; a constrained verification's grammar masks; the joint-acceptance draft probabilities).
ninfer_add_test(ninfer_qwen4_exp_program_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program_plan.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops ninfer_core)

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
# GPU, real artifact (skips without NINFER_TEST_QWEN4EXP_ARTIFACT): two lanes through the public
# Engine (--exact: every request's greedy ids against the same request on a one-lane Engine, with
# the context cache and a cancelled lane; --bench: requests in flight at once). Standalone: a script
# runs it by path with its own arguments.
ninfer_add_test(ninfer_qwen4_exp_lanes_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lanes_real.cpp"
  STANDALONE
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen4_exp_lanes_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_model_runtime ninfer_ops ninfer_core)
set_tests_properties(ninfer_qwen4_exp_engine_real_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): mtp_constrained_matches_plain, the
# engine test's --mtp-constrained mode. A greedy JSON-schema request with --mtp-constrained (MTP
# rounds, a grammar mask per verified position) gives the ids of plain constrained decoding.
add_test(NAME ninfer_qwen4_exp_mtp_constrained_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --mtp-constrained)
set_tests_properties(ninfer_qwen4_exp_mtp_constrained_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact (skips otherwise): partial_ram_residency_matches_full, the engine test's
# --partial-ram-exact mode. Greedy ids with --partial-ram-residency (RAM slots for the experts the
# VRAM cache does not hold, write-backs before a frame is reused) on a small expert cache equal those
# with every bank pinned, and at least one promotion copied a slot a write-back had filled.
add_test(NAME ninfer_qwen4_exp_partial_ram_residency_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --partial-ram-exact --expert-cache-mib 4096 --generate 256)
set_tests_properties(ninfer_qwen4_exp_partial_ram_residency_test PROPERTIES SKIP_RETURN_CODE 77)
# The same with MTP and --cpu-verify-misses on the partial Engine (the CPU reads RAM slots).
add_test(NAME ninfer_qwen4_exp_partial_ram_residency_cpu_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --partial-ram-exact --mtp 3 --cpu-verify-misses
          --expert-cache-mib 4096 --generate 256)
set_tests_properties(ninfer_qwen4_exp_partial_ram_residency_cpu_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact (skips otherwise): hot_embedding_matches_plain, the engine test's
# --hot-embedding-exact mode. Greedy ids with --hot-embedding-rows 8192 (the embedding's first rows in
# VRAM, the rest read from pinned RAM) equal those with the whole table in VRAM.
add_test(NAME ninfer_qwen4_exp_hot_embedding_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --hot-embedding-exact 8192)
set_tests_properties(ninfer_qwen4_exp_hot_embedding_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): gdn_head_map_matches_copies, the engine
# test's --gdn-head-map-exact mode. Greedy ids with --gdn-head-map (the GDN's 16 key heads read in
# place, MTP's replay records holding 16 key heads) equal those with the q/k copies, MTP rounds included.
add_test(NAME ninfer_qwen4_exp_gdn_head_map_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --gdn-head-map-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_gdn_head_map_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): moe_precast_matches_plain, the engine
# test's --moe-precast-exact mode. Greedy ids with --moe-precast-input (each routed-expert call casts
# its input to A8 once) equal those of the per-block casts, prefill, decode and MTP rounds included.
add_test(NAME ninfer_qwen4_exp_moe_precast_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --moe-precast-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_moe_precast_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): moe_gate_up_registers_matches_plain, the
# engine test's --moe-gate-up-registers-exact mode. Greedy ids with --moe-gate-up-registers (and the
# --moe-precast-input it needs) equal those of the staged gate/up kernel, MTP rounds included.
add_test(NAME ninfer_qwen4_exp_moe_gate_up_registers_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --moe-gate-up-registers-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_moe_gate_up_registers_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): moe_fused_route_matches_plain, the engine
# test's --moe-fused-route-exact mode. Greedy ids with --moe-fused-route (decode-width calls route and
# dispatch in one launch) equal those of the two-launch route, MTP rounds included.
add_test(NAME ninfer_qwen4_exp_moe_fused_route_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --moe-fused-route-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_moe_fused_route_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): cpu_verify_misses_matches_plain, the
# engine test's --cpu-verify-misses-exact mode. Greedy ids with --cpu-verify-misses (each
# verification's misses partly computed on the CPU worker team) equal those of the GPU-only
# verification, and so do the MTP rounds, drafts and acceptances; the CPU must serve at least one
# expert. A 4 GiB expert cache (about 2,700 of the 24,576 experts) keeps verification misses common.
add_test(NAME ninfer_qwen4_exp_cpu_verify_misses_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --cpu-verify-misses-exact --mtp 3 --expert-cache-mib 4096)
set_tests_properties(ninfer_qwen4_exp_cpu_verify_misses_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact (skips otherwise): prefill_cpu_share_matches_plain, the engine test's
# --prefill-cpu-share-exact mode. Greedy ids with --prefill-cpu-share 0.6 equal those of the GPU-only
# prefill for a 300-token prompt (one static call: each layer hands 3 / 5 of its misses, the
# fewest-column ones of at most 8, to the CPU worker team beside the GPU's staging) and 1,024- and
# 2,048-token prompts (one streamed call each, whose stream leaves each layer's predicted-cold
# experts out for the share). The CPU must serve at least one expert and a stream must leave one
# out. A 4 GiB expert cache keeps prefill misses common.
add_test(NAME ninfer_qwen4_exp_prefill_cpu_share_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --prefill-cpu-share-exact --expert-cache-mib 4096
          --prompt-lengths 300,1024,2048)
set_tests_properties(ninfer_qwen4_exp_prefill_cpu_share_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact (skips otherwise): ngram_flag_wait_matches_split, the engine test's
# --ngram-flag-wait-exact mode. Greedy ids with --ngram-flag-wait (each decode round one graph whose
# wait kernel holds the PLE layer back until the host has read the round's n-gram rows and raised a
# mapped flag) equal those of the two graphs around the host's read. The first runs plain rounds
# only (at least one must run as one graph); the second (with the MTP drafter, skips without it) adds
# the verifications, compares their rounds, drafts and acceptances, and needs at least one
# verification run as one graph.
add_test(NAME ninfer_qwen4_exp_ngram_flag_wait_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --ngram-flag-wait-exact)
set_tests_properties(ninfer_qwen4_exp_ngram_flag_wait_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ninfer_qwen4_exp_ngram_flag_wait_mtp_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --ngram-flag-wait-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_ngram_flag_wait_mtp_test PROPERTIES SKIP_RETURN_CODE 77)

# GPU, real artifact with the MTP drafter (skips otherwise): gdn_async_commit_matches_serial, the
# engine test's --gdn-async-commit-exact mode. Greedy ids with --gdn-async-commit (a continuing
# verification commit folds its GDN columns on a side stream; the next round's front waits for it)
# equal those of the fold on the device stream, and so do the MTP rounds, drafts and acceptances; at
# least one fold must run on the side stream.
add_test(NAME ninfer_qwen4_exp_gdn_async_commit_test
  COMMAND ninfer_tests ninfer_qwen4_exp_engine_real_test --gdn-async-commit-exact --mtp 3)
set_tests_properties(ninfer_qwen4_exp_gdn_async_commit_test PROPERTIES SKIP_RETURN_CODE 77)
