# Phantom-KV Integration Design

## Overview

Integrate "phantom graft" KV injection into NInfer — per-layer pre-computed K/V tensors that sit at the head of every active sequence, visible to attention but not part of the prompt. Each sequence carries its own graft spec; requests without a graft see nothing.

**PhantomKV upstream**: https://github.com/lordx64/phantom-kv  
**License**: MIT  
**Mechanism**: Learnable bank of per-layer K/V tensors (~MBs), trained as context that out-signals refusal circuits through ordinary attention. From the model's perspective, phantom KV is indistinguishable from conversation history.

---

## Architecture Mapping

```
Normal sequence KV: [prompt_tokens → pages 0..N]
Phantom sequence:   [graft_KV → pages 0..M-1] [prompt_tokens → pages M..N+M-1]
                                          ^^^^ text_kv_base set to M*64
```

### Where phantom lives

| Concept | Phantom analog | NInfer target |
|---|---|---|
| Pre-computed K/V bank | Per-layer K and V tensors (BF16) | New artifact type, loaded via existing converter pipeline |
| Per-layer splice | First M logical pages per layer | Block table entry 0..M-1 point to graft pages |
| Position offset | `text_kv_base = M * P` | Set early in prefill loop iteration |
| RoPE on tokens | Normal — graft sees `0..M*P-1`, prompt starts at `M*P` | No change needed |
| Attention masking | Causal over full `[graft | prompt | output]` | No change — kernel reads contiguous range |

### Capacity impact

Phantom pages consume **Main Text pool pages**, just like any other cached content. A 4B model with ~36 layers and 129 phantom slots per layer needs:

- Physical pages: `ceil(129 / 64) = 3 pages` per layer
- But since all layers share a common page-size, the 3 pages are the **same logical page IDs mapped into every layer's block table slice** — no, wait. Page IDs are pool-local, and each layer has its own planes within a single page-group. So 3 physical pages suffice for **all layers**.
- Per-sequence cost: 3 pages × ~512KB/page ≈ 1.5 MB/device for BF16

This is negligible compared to typical KV capacity (tens of GiB). Multiple sequences can share the same physical pages if they use identical grafts — reuse the existing immutability/Fork logic.

---

## Request-Level Opt-In

### Public API additions

**`include/ninfer/types.h`** — extend `ExecutionOptions`:

```cpp
// Graft specification for per-request KV cache injection.
// Empty means no graft. Non-empty specifies the named artifact file or session alias.
// When present, the engine reserves initial pages and writes graft KV before any prompt.
struct GraftSpec {
    std::string id;          // Unique identifier; maps to loaded graft artifact.
    // Optional: explicit slot count override. Default computed from artifact metadata.
    std::optional<std::uint32_t> slots;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    
    // NEW:
    std::optional<GraftSpec> graft;
};
```

No engine-wide configuration required. Every request independently opts in via `RequestOptions`.

### Graft resolution at submit time

When `submit(prepared_prompt, options)` is called:

1. If `options.graft.has_value()`, look up the graft by `id` in a registry
2. Registry entries map IDs to:
   - Path to serialized `.phantom` artifact file (loaded once at engine init)
   - Or a pre-loaded in-memory copy (for hot-swap scenarios)
3. Validate graft is compatible with the resident model (layer count, hidden dims, heads match)
4. Store the resolved reference in `RequestControl` for consumption during admission planning

---

## Implementation Stages

### Stage 1: Graft Artifact Format + Loader

**Goal**: Read pre-computed K/V tensors from a file and validate them against the resident model.

**Changes**:
- Define `.phantom` binary format: magic header, version, model-ID assertion, per-layer K/V tensor data with dtype/shape/dims metadata
- Create loader (`src/models/qwen3_5/program/storage/graft_loader.h/.cpp`)
- Loader validates layer count, head_dim, num_kv_heads match the current model's plan
- Cache loaded grafted tensors in Host pinned memory until needed (no H2D copy until first use)
- Register grafts via `Engine::register_graft(id, path)` — callable after engine construction

**Key types**:

```cpp
struct LayerGrafitData {
    std::int32_t layer_index;
    DType dtype;  // Always BF16 for now (matches training convention)
    std::int32_t head_dim;
    std::int32_t num_kv_heads;
    std::uint32_t slots; // Total phantom tokens across all layers
    
    // Device-ready tensors (pinned host staging for H2D transfer)
    Tensor k_tensor;  // [slots, head_dim] per layer
    Tensor v_tensor;  // [slots, head_dim] per layer
};

class LoadedGraft {
public:
    std::string model_id;    // Cross-check with engine
    std::vector<LayerGrafitData> layers;
    std::uint32_t max_slots; // Maximum slots across layers
};
```

**Placement**: Converters already handle artifact reading/writing. The graft loader is simpler — no quantization, no packing, just raw BF16 load. Could piggyback on existing deserialization infrastructure.

### Stage 2: Page Reservation + Write-on-Activation

**Goal**: When a request carrying a graft activates, reserve initial pages and write graft KV into them before prompt tokens begin.

**Primary hook**: Transaction start, in the prefill preparation flow inside `prefill.cpp`.

**Current flow** (simplified):
1. `staged.base` is set from checkpoint recovery or 0
2. Loop over prefill chunks: `staged.cursor` advances from `staged.base` to `staged.prompt_tokens`
3. Each chunk sets `schedule_state.text_kv_base = staged.cursor`
4. Prefill Op writes K/V at positions `cursor .. cursor+n`

**Modified flow**:
1. At activation (before `staged.base` is finalized):
   - If graft present: reserve N physical pages (N = ceil(max_slots / P)) from Main Text pool
   - These become "phantom pages" attached to the address space at indices 0..N-1
2. Set `staged.base = N * P` instead of 0
3. Copy graft K/tensors into reserved pages (single H2D burst)
4. Normal prefill proceeds from base onwards

**Transaction-level changes** (`program_impl.cpp` / transaction flow):

In the resource plan execution path, before normal prefill begins:

```
if (request has graft):
    // 1. Reserve extra pages beyond what prompt needs
    auto graft_pages = reserve_device_kv_page_bundle({
        {.pool = &text_cache.page_pool(), .pages = num_graft_pages}
    });
    
    // 2. Assign to logical pages 0..num_graft_pages-1
    assign_logical_pages(graft_pages);
    
    // 3. H2D copy of graft K/V into device pages
    for each layer i:
        write_graft_layer(layer_i, graft_pages, stream);
    
    // 4. Map into block table row at indices 0..num_graft_pages-1
    publish_block_table_slice(/*begin=*/0, /*count=*/num_graft_pages);
    
    // 5. Advance base past graft region
    staged.base = num_graft_pages * P;
```

**Critical detail — block table publication timing**: The block table must be fully populated for all accessed logical indices before the first prefill chunk launches. Phantom pages are known at activation time, so they're published alongside prompt-derived pages.

**`kv_store.h` additions**:

```cpp
// Reserve initial pages for graft injection. Called during transaction start
// before any prefill. Returns the assigned logical page handles.
std::vector<LogicalKVPageHandle> reserve_initial_pages_for_graft(
    KVAddressSpaceHandle address,
    std::uint32_t page_count,
    DeviceKVPageReservation&& reservation);
```

### Stage 3: Attention Consumption Path

**Goal**: Ensure attention kernels correctly attend over `[graft_kv | prompt_kv | ...]` when computing logits.

**Analysis**: The existing attention consumer (`PagedKVLayerView`) takes a block table as input. For position `p`:

```cpp
logical_block = p >> 6;      // p / 64
page_offset   = p & 63;     // p % 64  
physical_page = block_table[logical_block];
```

If block table index 0 points to a graft page, and position `p=5` accesses `block_table[0]`, the kernel naturally reads graft KV at positions 0..63. **No kernel changes needed** — this is pure block-table plumbing.

**RoPE positions**: Position tensors passed to attention will include `0..M*P-1` for graft positions. RoPE computes normally — there's nothing special about graft positions, they're just early positions.

**Verification**: After Stage 2, add a correctness test:

```
Test: graft_identity_verification
Setup: Single-graft model, simple graft (e.g., all-zeros K with specific V pattern)
Action: Submit a single-token prompt with graft enabled
Assert: Output differs from no-graft baseline (proves graft was read)
```

### Stage 4: Mid-Session Swap (Hot-Swap Pills)

**Goal**: Allow swapping grafts mid-generation without restarting inference.

This maps directly to NInfer's existing Move/dematerialize/lease primitives:

```
Swap operation (phatom_kvs.h):
    1. Validate new graft is compatible with active sequence's model
    2. Deactivate old graft pages (dematerialize leases, release from address space)
    3. Materialize new graft pages into the same logical page IDs (or new ones)
    4. Publish updated block table entries 0..N-1
    5. Note: positions of prompt/output shift relative to graft boundary
       This means existing decoded tokens' attention computation doesn't change,
       but future tokens will attend over new graft.
```

**API**:

```cpp
// Engine method — swaps graft for an active sequence identified by handle.
void Engine::swap_graft(GenerationHandle handle, const std::string& new_graft_id);
```

**Implementation challenge**: When swapping mid-decode, the decode batch still has a valid `execution_frontier` that assumed the old graft was at positions 0..M. New generations continue normally because `text_kv_valid` tracks actual written pages, and the new graft occupies the same early pages. The block table update is atomic within one transaction boundary.

### Stage 5: Retention Policy Integration

**Problem**: Current eviction/pressure planning treats all KV pages equally. A graft page should be protected from eviction.

**Solution**: Tag initial pages (those with logical index < num_graft_pages) as `protected` in `LogicalKVPage` metadata. Pressure planner skips protected pages during victim selection.

```cpp
class LogicalKVPage {
    // ... existing fields ...
    
    [[nodiscard]] bool is_protected() const noexcept {
        return is_graft_page || is_checkpoint_protected();
    }
    
private:
    bool is_graft_page = false;  // Set during reservation
};
```

**Pressure planner change** (`materialization_planner.h` / pressure planning code): Skip pages where `is_graft_page` during eviction search. This is a small filter in the victim enumeration loop.

---

## File Change Summary

| File | Change |
|---|---|
| `include/ninfer/types.h` | Add `GraftSpec`, `ExecutionOptions.graft`, `Engine::register_graft()` |
| `src/core/paged_kv_cache.h` | Extend `LogicalKVPage` with `is_graft_page` flag |
| `src/models/qwen3_5/program/storage/graft_loader.h/.cpp` | **New**: Graft artifact format + loader |
| `src/models/qwen3_5/program/storage/graft_storage.h/.cpp` | **New**: In-engine graft registry (ID → LoadedGraft) |
| `src/models/qwen3_5/program/storage/kv_store.h` | Add `reserve_initial_pages_for_graft()`, `mark_as_graft()` |
| `src/models/qwen3_5/program/prefill.cpp` | Hook: check graft flag, reserve pages, set base offset, H2D copy |
| `src/models/qwen3_5/program/program_impl.cpp` | Hook: grant pages in transaction, manage graft lifecycle |
| `src/models/qwen3_5/frontend/prepared_prompt.h` | Expose graft spec from request |
| `src/models/qwen3_5/program/planning/materialization_planner.cpp` | Protect graft pages from eviction |
| `tests/unit/test_graft_integration.cpp` | **New**: Correctness + basic perf tests |
| `docs/design/phantom-kv-integration.md` | **New**: This document |

**Estimated diff**: ~1200 lines of new code + ~200 lines modified across existing files.

---

## Risk Assessment

### Low risk
- Block table plumbing: NInfer already supports arbitrary page-to-position mapping. Phantom pages are just early pages.
- Attention kernels: Unchanged. They read whatever the block table says.
- H2D copy: Standard operation, used everywhere for state restore.

### Medium risk
- Page reservation under tight capacity: Extra pages reduce effective KV capacity for prompt tokens. Need to verify capacity planning accounts for this. Mitigation: limit graft to fixed max (e.g., 3 pages), document that users with very tight capacity should disable graft.
- Mid-session swap: Requires careful transaction ordering to avoid stale reads. Use existing reservation protocol.

### High risk
- Training quality: The hardest part is generating effective grafts. PhantomKV's training pipeline produces decent results on Qwen3-4B, but adapting it to your specific model/target config requires experimentation. This is research work, not C++ work.
- Performance regression: Each graft adds attention over M extra positions. With M=129, that's ~129 extra KV tokens attended per forward pass. On sm_86, this is measurable but bounded — roughly equivalent to attending over a ~2 token prompt increase. Not significant.

---

## Testing Plan

| Test | Scope | Method |
|---|---|---|
| `graft_load_validation` | Loader | Invalid format/model-id rejects gracefully |
| `graft_write_correctness` | Page write | Bit-identical round-trip: write graft → copy back → compare |
| `graft_attention_effect` | End-to-end | Same prompt, with/without graft → different output provenance |
| `graft_no_regression` | End-to-end | Without graft → bit-identical to mainline (no phantom code path entered) |
| `graft_swap_mid_decode` | Lifecycle | Activate graft → generate 100 tokens → swap → generate 100 more → verify continuity |
| `graft_capacity_bound` | Resource planning | Tight capacity with graft → admission rejection works correctly |
| `graft_prefixed_to_existing` | Prefix caching | Checkpoint contains graft prefix → fork/new session inherits correctly |

---

## Open Questions

1. **Per-layer slot count**: Should all layers share the same slot count (simplest), or support variable slots per layer? Start with uniform. PhantomKV trains uniform banks.

2. **Host resident grafts**: Keep loaded graft tensors in host pinned memory indefinitely? Or only stage per-request? For multiple sequences using the same graft, staging once and reusing would be better. Suggestion: pin loaded grafts in host arena; H2D copy per-sequence on first activation.

3. **Thread safety of swap**: Can `swap_graft()` be called from the thread that owns the GenerationHandle, even while another lane is decoding? Answer: Yes, if it goes through the standard transaction mechanism. The swap becomes a zero-tokens resource transition that updates the block table.

4. **Quantized grafts**: PhantomKV trains in BF16. Should we support INT8/NVFP4 packed formats? Decision: defer. BF16 graft tensors are tiny (a few KB per layer) — quantizing saves microseconds of H2D transfer, not meaningful bytes.
