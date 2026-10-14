#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::context_blocks(SequenceHandle sequence) const noexcept {
    if (!context_transaction_ || !valid_sequence(sequence)) { return false; }
    const auto& tx = *context_transaction_;
    return tx.kind == ContextOperationKind::Bind && (tx.text_view || tx.backend_view) &&
           sequences[ContractAccess::lane(sequence).value].kv == tx.source_history;
}

void ProgramImpl::enqueue_state_backup(ContextTransaction& tx) {
    if (!tx.state_transfer) { return; }
    tx.transfers.push_back(state_transfer_requirement(
        state_images->host_layout(), runtime::ContextTransferDirection::DeviceToHost));
    start_context_transfer_timer(runtime::ContextResourceClass::State);
    state_store->enqueue_device_to_host(*tx.state_transfer, transfer_streams);
    stop_context_transfer_timer(runtime::ContextResourceClass::State);
}

void ProgramImpl::copy_local_for_context(ContextTransaction& tx, std::int32_t source,
                                         std::int32_t destination) {
    tx.transfers.push_back(state_transfer_requirement(
        state_images->host_layout(), runtime::ContextTransferDirection::DeviceToDevice, true));
    start_context_transfer_timer(runtime::ContextResourceClass::State);
    state_images->copy_dflash_local(source, destination, transfer_streams);
    stop_context_transfer_timer(runtime::ContextResourceClass::State);
}

void ProgramImpl::copy_context_tail(ContextTransaction& tx, LogicalKVPageStore& pages,
                                    DeviceKVPageHandle source, DeviceKVPageHandle destination,
                                    runtime::ContextResourceClass resource) {
    if (tx.layered_restore && resource == runtime::ContextResourceClass::MainKV) {
        // Its source may be a page still landing: copy it layer by layer behind each layer's
        // restore (enqueue_layered_restore).
        tx.layered_tail.emplace(source, destination);
        ++tx.operations.partial_tail_cow_pages;
        return;
    }
    start_context_transfer_timer(resource);
    const auto work = pages.physical_pool().copy_page(source, destination, transfer_streams);
    stop_context_transfer_timer(resource);
    tx.transfers.push_back(kv_transfer_requirement(
        resource, runtime::ContextTransferDirection::DeviceToDevice, 1, work));
    ++tx.operations.partial_tail_cow_pages;
}

void ProgramImpl::enqueue_context_transfers(ContextTransaction& tx) {
    // A Bind restoring Main KV from the Host lands it layer by layer after the binding completes:
    // the binding waits only for the prelude (the state image and backend KV), and the lane's first
    // prefill pass waits for each full-attention layer's copies just before that layer
    // (enqueue_layered_restore). Replays and split models keep the whole restore inside the binding.
    tx.layered_restore =
        tx.kind == ContextOperationKind::Bind && !tx.resume && transfer_streams.size() == 1 &&
        std::any_of(tx.kv_transfers.begin(), tx.kv_transfers.end(), [](const KVTransfer& t) {
            return t.restore && t.resource == runtime::ContextResourceClass::MainKV &&
                   !t.logical.empty();
        });
    // Runs of pages contiguous in one Host extent, each copied as one transfer.
    const auto host_runs = [&](KVTransfer& transfer, auto&& copy) {
        std::size_t begin = 0;
        while (begin < transfer.logical.size()) {
            const auto first = transfer.pages->host_replica(transfer.logical[begin]);
            std::size_t end  = begin + 1;
            while (end < transfer.logical.size()) {
                const auto next = transfer.pages->host_replica(transfer.logical[end]);
                if (next.extent != first.extent ||
                    next.page_offset != first.page_offset + end - begin) {
                    break;
                }
                ++end;
            }
            copy(host_kv_extents->view(first.extent)
                     .subview(first.page_offset, static_cast<std::uint32_t>(end - begin)),
                 std::span<const DeviceKVPageHandle>(transfer.physical).subspan(begin, end - begin));
            begin = end;
        }
    };
    for (const auto resource :
         {runtime::ContextResourceClass::MainKV, runtime::ContextResourceClass::BackendKV}) {
        bool started = false;
        bool restore = false;
        TransferWork total;
        std::uint32_t page_count = 0;
        for (auto& transfer : tx.kv_transfers) {
            if (transfer.resource != resource || transfer.logical.empty()) { continue; }
            if (tx.layered_restore && resource == runtime::ContextResourceClass::MainKV) {
                // Planned now, while the source pages are pinned by this transaction; copied after
                // the prelude has landed.
                host_runs(transfer, [&](HostKVAllocationConstView host,
                                        std::span<const DeviceKVPageHandle> destination) {
                    tx.layered_runs.push_back(
                        {.pages       = transfer.pages,
                         .host        = host,
                         .destination = {destination.begin(), destination.end()}});
                });
                tx.layered_pages += static_cast<std::uint32_t>(transfer.logical.size());
                continue;
            }
            if (!started) {
                start_context_transfer_timer(resource);
                restore = transfer.restore;
                started = true;
            } else if (restore != transfer.restore) {
                throw std::logic_error("one transfer batch mixes directions within a typed pool");
            }
            TransferWork work;
            if (transfer.restore) {
                host_runs(transfer, [&](HostKVAllocationConstView host,
                                        std::span<const DeviceKVPageHandle> destination) {
                    const auto part = transfer.pages->physical_pool().copy_from_host(
                        host, destination, transfer_streams);
                    work.payload_bytes += part.payload_bytes;
                    work.copy_operations += part.copy_operations;
                });
            } else if (transfer.host_destination) {
                const auto host = host_kv_extents->writable_view(*transfer.host_destination);
                work = transfer.pages->physical_pool().copy_to_host(transfer.physical, host,
                                                                    transfer_streams);
                if (tx.kind == ContextOperationKind::Pause ||
                    tx.kind == ContextOperationKind::Demote) {
                    tx.operations.pressure_spill_pages += transfer.physical.size();
                }
            }
            total.payload_bytes += work.payload_bytes;
            total.copy_operations += work.copy_operations;
            page_count += static_cast<std::uint32_t>(transfer.logical.size());
        }
        if (!started) { continue; }
        stop_context_transfer_timer(resource);
        tx.transfers.push_back({.resource   = resource,
                                .direction  = restore
                                                  ? runtime::ContextTransferDirection::HostToDevice
                                                  : runtime::ContextTransferDirection::DeviceToHost,
                                .units      = total.payload_bytes,
                                .page_count = page_count,
                                .work       = total});
    }
    // A layered Bind always waits for its prelude (possibly empty), then plans the layers.
    tx.submitted = tx.layered_restore || !tx.transfers.empty();
    if (tx.submitted) { context_completion_.record(transfer_streams); }
}

void ProgramImpl::enqueue_layered_restore(ContextTransaction& tx) {
    const auto layers = decoder->text_kv.layers();
    const auto planes = text_kv_pages->physical_pool().plane_count();
    if (layers == 0 || planes % layers != 0) {
        throw std::logic_error("Main KV planes are not a whole number per layer");
    }
    const std::size_t per_layer = planes / layers;
    // The landing is built privately and published into `landing_restores_` only once every event
    // is recorded, so retire_landing / abort_context / the destructor never see a partial one. The
    // slot is reserved first so publishing cannot throw after the copies were queued.
    landing_restores_.reserve(landing_restores_.size() + 1);
    LandingRestore landing{.lane = tx.lane, .timer = CudaEventTimer(device, transfer_streams[0])};
    landing.page_count = tx.layered_pages;
    if (tx.source_history) { landing.sources.push_back(tx.source_history); }
    // Every Host byte the gather kernels will read, for the release fence. A run whose view cannot
    // be measured is recorded as unknown, which makes any release wait for this landing.
    landing.host_ranges.reserve(tx.layered_runs.size());
    for (const auto& run : tx.layered_runs) {
        HostRange read;
        if (run.host.valid()) {
            try {
                read = {run.host.data(), static_cast<std::size_t>(run.host.page_count()) *
                                             run.host.layout().page_stride};
            } catch (...) { read = {}; }
        }
        landing.host_ranges.push_back(read);
    }
    landing.events.reserve(layers);
    try {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            if (spare_restore_events_.empty()) {
                cudaEvent_t event = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                try {
                    spare_restore_events_.push_back(event);
                } catch (...) {
                    (void)cudaEventDestroy(event);
                    throw;
                }
            }
            landing.events.push_back(spare_restore_events_.back());
            spare_restore_events_.pop_back();
        }
        // Describe every layer's copies first, then run each layer as gather kernels reading their
        // descriptors from a pinned buffer the landing keeps until it has landed: the layer's
        // restore, then its share of a private tail, whose source may be a page that restore writes.
        // The tail's Device source needs no pin here: its readers are on the transfer stream, and
        // every other writer of a page recycled from it is ordered behind this landing (the lane's
        // prefill waits per layer, all other compute work on the last layer: take_restore_layers /
        // order_after_restores), as are all later transfers on the same stream.
        constexpr std::size_t kPieceBytes = std::size_t{64} << 10;
        std::vector<GatherCopy> copies, layer_copies;
        // Per layer: [restore begin, tail begin, tail end) in `copies`.
        std::vector<std::array<std::size_t, 3>> segments(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const std::size_t begin = layer * per_layer, end = begin + per_layer;
            segments[layer][0] = copies.size();
            layer_copies.clear();
            for (const auto& run : tx.layered_runs) {
                const auto work = run.pages->physical_pool().describe_from_host(
                    run.host, run.destination, begin, end, layer_copies);
                landing.work.payload_bytes += work.payload_bytes;
                landing.work.copy_operations += work.copy_operations;
            }
            split_gather_copies(layer_copies, kPieceBytes);
            copies.insert(copies.end(), layer_copies.begin(), layer_copies.end());
            segments[layer][1] = copies.size();
            if (tx.layered_tail) {
                layer_copies.clear();
                const auto work = text_kv_pages->physical_pool().describe_page_copy(
                    tx.layered_tail->first, tx.layered_tail->second, begin, end, layer_copies);
                landing.tail_work.payload_bytes += work.payload_bytes;
                landing.tail_work.copy_operations += work.copy_operations;
                split_gather_copies(layer_copies, kPieceBytes);
                copies.insert(copies.end(), layer_copies.begin(), layer_copies.end());
            }
            segments[layer][2] = copies.size();
        }
        const std::size_t bytes = copies.size() * sizeof(GatherCopy);
        for (auto it = spare_descriptor_buffers_.begin(); it != spare_descriptor_buffers_.end();
             ++it) {
            if (it->size() >= bytes) {
                landing.descriptors.emplace(std::move(*it));
                spare_descriptor_buffers_.erase(it);
                break;
            }
        }
        if (!landing.descriptors) {
            landing.descriptors.emplace(std::max<std::size_t>(bytes, std::size_t{1} << 20));
        }
        auto* staged = static_cast<GatherCopy*>(landing.descriptors->data());
        std::copy(copies.begin(), copies.end(), staged);
        landing.timer.start();
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const auto [restore, tail, after] = segments[layer];
            gather_copies(std::span<const GatherCopy>(staged + restore, tail - restore),
                          transfer_streams[0]);
            gather_copies(std::span<const GatherCopy>(staged + tail, after - tail),
                          transfer_streams[0]);
            // The last layer's event follows the timer's stop, so a reader that has waited for
            // every layer can read the elapsed time.
            if (layer + 1 == layers) { landing.timer.record_stop(); }
            CUDA_CHECK(cudaEventRecord(landing.events[layer], transfer_streams[0]));
        }
    } catch (...) {
        // Queued gather kernels may still read the descriptors: drain the stream, then give the
        // events and the buffer back. The transaction still owns every source page and run.
        synchronize_transfer_streams();
        recycle_landing_resources(landing);
        throw;
    }
    landing_restores_.push_back(std::move(landing));
    tx.layered_runs.clear();
    tx.layered_tail.reset();
}

void ProgramImpl::recycle_landing_resources(LandingRestore& landing) noexcept {
    // Capacity is reserved for both pools before either is touched, so the transfers below cannot
    // throw half way: a handle is then in exactly one place, a pool or destroyed, never both.
    try {
        spare_restore_events_.reserve(spare_restore_events_.size() + landing.events.size());
        if (landing.descriptors) {
            spare_descriptor_buffers_.reserve(spare_descriptor_buffers_.size() + 1);
        }
    } catch (...) {
        for (const auto event : landing.events) { (void)cudaEventDestroy(event); }
        landing.events.clear();
        landing.descriptors.reset(); // the pinned buffer is freed instead of pooled
        return;
    }
    spare_restore_events_.insert(spare_restore_events_.end(), landing.events.begin(),
                                 landing.events.end());
    landing.events.clear();
    if (landing.descriptors) {
        spare_descriptor_buffers_.push_back(std::move(*landing.descriptors));
        landing.descriptors.reset();
    }
}

std::span<const cudaEvent_t> ProgramImpl::take_restore_layers(std::uint32_t lane) {
    std::span<const cudaEvent_t> out;
    for (auto& landing : landing_restores_) {
        if (landing.ordered) { continue; }
        if (landing.lane == lane) {
            out = landing.events;
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(compute_streams[0], landing.events.back(), 0));
        }
        landing.ordered = true;
    }
    return out;
}

void ProgramImpl::order_after_restores() {
    for (auto& landing : landing_restores_) {
        if (landing.ordered) { continue; }
        CUDA_CHECK(cudaStreamWaitEvent(compute_streams[0], landing.events.back(), 0));
        landing.ordered = true;
    }
}

void ProgramImpl::land_restores() {
    for (const auto& landing : landing_restores_) {
        CUDA_CHECK(cudaEventSynchronize(landing.events.back()));
    }
}

void ProgramImpl::land_restores_overlapping(HostRange range) noexcept {
    // Only a landing that reads the bytes being released can be hurt by their release. Waiting
    // for the rest stalls the Engine on the GPU's current prefill chunk for nothing (a landing
    // completes as its lane's first prefill pass advances, which is about a second per chunk).
    // A landing with no recorded range, or a release of unknown bytes, still waits.
    for (const auto& landing : landing_restores_) {
        if (landing.events.empty()) { continue; }
        if (!release_must_wait_for(landing.host_ranges, range)) {
            ++release_fence_skips_;
            continue;
        }
        const auto begin = std::chrono::steady_clock::now();
        (void)cudaEventSynchronize(landing.events.back());
        release_fence_wait_ns_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count());
        ++release_fence_waits_;
    }
}

void ProgramImpl::complete_landing(std::uint32_t lane) {
    for (auto it = landing_restores_.begin(); it != landing_restores_.end(); ++it) {
        if (it->lane != lane) { continue; }
        CUDA_CHECK(cudaEventSynchronize(it->events.back()));
        const double elapsed_ns = static_cast<double>(it->timer.elapsed_ms()) * 1'000'000.0;
        landed_transfers_.push_back(runtime::ContextTransferObservation{
            .resource   = runtime::ContextResourceClass::MainKV,
            .direction  = runtime::ContextTransferDirection::HostToDevice,
            .units      = it->work.payload_bytes,
            .page_count = it->page_count,
            .work       = it->work,
            .elapsed_ns = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5)),
        });
        if (it->tail_work.payload_bytes != 0) {
            // A private tail copied layer by layer between the restore's copies has no interval
            // of its own: its bytes are reported, its time is inside the restore's.
            landed_transfers_.push_back(runtime::ContextTransferObservation{
                .resource   = runtime::ContextResourceClass::MainKV,
                .direction  = runtime::ContextTransferDirection::DeviceToDevice,
                .units      = it->tail_work.payload_bytes,
                .page_count = 1,
                .work       = it->tail_work,
                .elapsed_ns = 1,
            });
        }
        recycle_landing_resources(*it);
        // The copies have landed: the source history may now be released (and its Host pages
        // reclaimed) after the landing is out of the list, so the release fence finds nothing to wait for.
        auto sources = std::move(it->sources);
        landing_restores_.erase(it);
        sources.clear();
        return;
    }
}

void ProgramImpl::retire_landing(std::uint32_t lane) noexcept {
    for (auto it = landing_restores_.begin(); it != landing_restores_.end();) {
        if (it->lane != lane) {
            ++it;
            continue;
        }
        // The lane's pages may return to the pool: nothing may reuse them before the copies land.
        (void)cudaEventSynchronize(it->events.back());
        recycle_landing_resources(*it);
        auto sources = std::move(it->sources);
        it           = landing_restores_.erase(it);
        sources.clear(); // landed: release the source history now (see complete_landing)
    }
}

void ProgramImpl::publish_context_transfers(ContextTransaction& tx) {
    if (tx.state_transfer) {
        state_store->publish_transfer(std::move(*tx.state_transfer), tx.preserve_state_device);
        tx.state_transfer.reset();
    }
    for (auto& transfer : tx.kv_transfers) {
        if (transfer.restore) {
            for (const auto page : transfer.logical) {
                transfer.pages->publish_device_replica(page);
            }
            transfer.logical.clear();
        } else if (transfer.host_destination) {
            (void)host_kv_extents->publish(std::move(*transfer.host_destination));
            transfer.host_destination.reset();
        }
        if (transfer.drop_device) {
            for (const auto page : transfer.logical) {
                if (!transfer.pages->drop_device_replica(page)) {
                    throw std::logic_error("published Host copy still has a Device reader");
                }
            }
        }
    }
}

ContextProgress ProgramImpl::poll_context(runtime::CancellationFlagView cancellation) {
    if (!context_transaction_) { throw std::logic_error("there is no context operation"); }
    auto& tx = *context_transaction_;
    ContextProgress out{.kind = tx.kind};
    // Bind has two transfer phases. Advance both in this call when their real events
    // are already complete; an unready event yields without synchronizing or spinning.
    for (std::uint32_t phase = 0; phase < 2; ++phase) {
        if (tx.submitted && !context_completion_.ready()) { return out; }
        for (const auto& transfer : tx.transfers) {
            tx.observations.push_back(context_transfer_observation(
                transfer.resource, transfer.direction, transfer.work, transfer.page_count,
                transfer.resource == runtime::ContextResourceClass::State ? transfer.units : 0));
        }
        tx.transfers.clear();
        if (tx.kind == ContextOperationKind::Pause && tx.paused && tx.paused->impl_) {
            out.request_timings     = tx.paused->impl_->control.timings;
            out.request_speculative = tx.paused->impl_->control.speculative_stats;
        }
        if (cancellation.requested()) {
            out.transfers  = std::move(tx.observations);
            out.operations = tx.operations;
            abort_context();
            out.complete = true;
            return out;
        }
        if (!tx.binding_prepared) {
            publish_context_transfers(tx);
            if (tx.kind == ContextOperationKind::Bind) {
                prepare_binding(tx);
                tx.binding_prepared = true;
                out.advanced        = true;
                continue;
            }
        }
        if (tx.kind == ContextOperationKind::Bind) {
            complete_binding(tx, out);
        } else if (tx.kind == ContextOperationKind::Capture) {
            // Capture publication is implemented with the matching state/KV reservation in
            // capture.cpp.
            out.captured_checkpoints.reserve(tx.capture_points.size());
            publish_capture(tx);
            for (const auto& [handle, role] : tx.capture_points) {
                out.captured_checkpoints.push_back(handle);
            }
        } else if (tx.kind == ContextOperationKind::Pause) {
            out.paused.emplace(complete_pause(tx));
        }
        if (tx.source) { --checkpoints[tx.source->index].pins; }
        for (const auto point : tx.carried_pins) { --checkpoints[point.index].pins; }
        tx.carried_pins.clear();
        out.complete = out.published = true;
        out.transfers                = std::move(tx.observations);
        out.operations               = tx.operations;
        context_transaction_.reset();
        return out;
    }
    return out;
}

void ProgramImpl::abort_context() noexcept {
    if (!context_transaction_) { return; }
    auto& tx = *context_transaction_;
    try {
        // Enqueue can fail before recording the completion event. Drain the transfer stream
        // itself so every submitted reader has retired before releasing its source/destination.
        // CUDA failure is already fatal to the Engine; it must not prevent CPU ownership cleanup.
        synchronize_transfer_streams();
        if (tx.state_transfer) {
            state_store->abort_transfer(std::move(*tx.state_transfer));
            tx.state_transfer.reset();
        }
        for (auto& transfer : tx.kv_transfers) {
            if (transfer.restore) {
                for (const auto page : transfer.logical) {
                    transfer.pages->abort_device_replica(page, *transfer.device_reservation);
                }
            }
            transfer.host_destination.reset();
        }
        tx.text_fork.reset();
        tx.backend_fork.reset();
        tx.text_activation.reset();
        tx.backend_activation.reset();
        tx.text_view.reset();
        tx.backend_view.reset();
        if (tx.kind == ContextOperationKind::Bind && tx.adopted) {
            release_sequence_state(sequences[tx.lane]);
            release_sequence_kv(sequences[tx.lane]);
            tx.reserved_state.reset();
        }
        if (tx.reserved_kv) {
            const auto unwind_address = [&](KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address, bool borrowed,
                                            std::uint32_t frontier) {
                if (!borrowed) {
                    if (!addresses.release_after_deactivate(address)) { std::terminate(); }
                } else if (addresses.active(address)) {
                    addresses.deactivate(address);
                    addresses.set_checkpoint_requirement(address, frontier);
                }
            };
            if (tx.reserved_kv->backend) {
                unwind_address(*backend_kv_addresses, *tx.reserved_kv->backend, tx.borrow_backend,
                               tx.backend_frontier);
            }
            unwind_address(*text_kv_addresses, tx.reserved_kv->text, tx.borrow_text,
                           tx.reuse_frontier);
            tx.reserved_kv.reset();
        }
        if (tx.kind == ContextOperationKind::Bind) {
            // Any already-installed state binding is still private to this unpublished lane.
            if (sequences[tx.lane].state.fork_pending) {
                state_store->abort_fork(sequences[tx.lane].state.read,
                                        sequences[tx.lane].state.write);
            }
            sequences[tx.lane].state = {};
            sequences[tx.lane].kv.reset();
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        }
        if (tx.kind == ContextOperationKind::Capture) {
            auto& state = sequences[tx.lane];
            refresh_history_requirements(state.kv);
            if (state_store->role(state.state.read) == StateImageRole::CheckpointImmutable &&
                !state_store->checkpoint_references(state.state.read)) {
                state_store->thaw(state.state.read);
            }
        }
        if (tx.kind == ContextOperationKind::Pause) {
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        }
        if (tx.reserved_state && !tx.borrow_state) {
            (void)state_store->release(*tx.reserved_state);
        }
        if (tx.source) { --checkpoints[tx.source->index].pins; }
        for (const auto point : tx.carried_pins) { --checkpoints[point.index].pins; }
        for (const auto& [from, to] : tx.carried_clones) {
            if (valid_checkpoint(to)) {
                (void)release_checkpoint(to);
            } else {
                checkpoints[to.index].reserved = false;
            }
        }
        for (const auto& [handle, role] : tx.capture_points) {
            if (!valid_checkpoint(handle)) { checkpoints[handle.index].reserved = false; }
        }
        context_transaction_.reset();
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen3_5::detail
