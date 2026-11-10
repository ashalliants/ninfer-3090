// Public-contract qualification of ops::numeric_tripwire_scan against the naive host oracle in
// numeric_tripwire_oracle.h (plain loop in double, element decoding written out bit by bit; no
// kernel code is reused). The record must match the oracle exactly: which site owns it, the lowest
// offending element and its bit pattern, and the total count. Written without a GPU to run it on;
// the host oracle itself is checked by tests/runtime/test_numeric_tripwire.cpp.
#include "ninfer/ops/numeric_tripwire.h"
#include "ops/numeric_tripwire_oracle.h"
#include "ops/op_tester.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kInf   = std::numeric_limits<float>::infinity();
constexpr float kNan   = std::numeric_limits<float>::quiet_NaN();
constexpr float kLimit = 65504.0F;

struct Encoded {
    std::vector<std::uint8_t> bytes;
    std::vector<float> widened; // the exact FP32 value of every stored element
};

Encoded encode(DType dtype, const std::vector<float>& input) {
    Encoded out;
    out.widened.resize(input.size());
    if (dtype == DType::FP32) {
        out.bytes.resize(input.size() * 4);
        for (std::size_t i = 0; i < input.size(); ++i) {
            std::memcpy(out.bytes.data() + i * 4, &input[i], 4);
            out.widened[i] = input[i];
        }
        return out;
    }
    out.bytes.resize(input.size() * 2);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::uint16_t bits = dtype == DType::FP16 ? tripwire_float_to_half_bits(input[i])
                                                        : tripwire_float_to_bf16_bits(input[i]);
        std::memcpy(out.bytes.data() + i * 2, &bits, 2);
        out.widened[i] = dtype == DType::FP16 ? tripwire_half_bits_to_float(bits)
                                              : tripwire_bf16_bits_to_float(bits);
    }
    return out;
}

const char* name(DType dtype) {
    return dtype == DType::FP16 ? "fp16" : dtype == DType::BF16 ? "bf16" : "fp32";
}

// A device record plus the oracle's expectation of it, accumulated over the calls made so far.
class Probe {
public:
    Probe() : record_(ninfer::ops::kNumericTripwireRecordBytes), tensor_(record_.data(), DType::U8, {16}) {
        clear();
    }

    void clear() {
        record_.fill(0);
        expected_ = {};
    }

    Tensor& tensor() { return tensor_; }
    TripwireOracleRecord& expected() { return expected_; }

    // Runs one scan and applies the same scan to the oracle record.
    int scan(const std::string& label, DType dtype, const std::vector<float>& input, float limit,
             std::uint32_t site, cudaStream_t stream) {
        const Encoded encoded = encode(dtype, input);
        GuardedDeviceBuffer values(encoded.bytes.size());
        values.copy_from_host(encoded.bytes.data(), encoded.bytes.size());
        Tensor tensor(values.data(), dtype, {static_cast<std::int32_t>(input.size())});
        ops::numeric_tripwire_scan(tensor, site, limit, tensor_, stream);
        cuda_synchronize(stream);
        expected_.apply(site, tripwire_oracle_scan(encoded.widened, limit));

        int failures = check(label);
        std::vector<std::uint8_t> after(encoded.bytes.size());
        values.copy_to_host(after.data(), after.size());
        if (after != encoded.bytes) {
            std::cerr << label << ": the scan modified its input\n";
            ++failures;
        }
        failures += values.verify_guards(label + " values");
        return failures;
    }

    int check(const std::string& label) {
        ops::NumericTripwireRecord actual{};
        record_.copy_to_host(&actual, sizeof(actual));
        int failures = 0;
        const auto fail = [&](const char* what) {
            std::cerr << label << ": " << what << '\n';
            ++failures;
        };
        const auto finding = ops::decode_numeric_tripwire_record(actual);
        if (expected_.site_plus_one == 0) {
            if (actual.site_plus_one != 0 || actual.best != 0) { fail("record claimed without a finding"); }
        } else if (!finding) {
            fail("the oracle saw a bad value and the record is empty");
        } else {
            if (actual.site_plus_one != expected_.site_plus_one) { fail("wrong owning site"); }
            if (finding->element_index != expected_.best_index) { fail("not the lowest offending index"); }
            if (std::bit_cast<std::uint32_t>(finding->value) !=
                std::bit_cast<std::uint32_t>(expected_.best_value)) {
                fail("wrong recorded value bits");
            }
        }
        if (actual.count != expected_.count) {
            std::cerr << label << ": count " << actual.count << ", oracle " << expected_.count << '\n';
            ++failures;
        }
        failures += record_.verify_guards(label + " record");
        return failures;
    }

private:
    GuardedDeviceBuffer record_;
    Tensor tensor_;
    TripwireOracleRecord expected_;
};

int run_dtype_cases(DType dtype, cudaStream_t stream) {
    int failures = 0;
    const std::string tag = name(dtype);
    const auto run = [&](const char* label, std::vector<float> input, float limit = kLimit,
                         std::uint32_t site = 4) {
        Probe probe;
        failures += probe.scan(tag + " " + label, dtype, input, limit, site, stream);
    };
    const auto filled = [](std::size_t n) {
        std::vector<float> v(n);
        fill_uniform(v, 1234U + static_cast<std::uint32_t>(n), -8.0F, 8.0F);
        return v;
    };

    run("clean 4096", filled(4096));
    {
        auto v = filled(4096);
        v[777] = kNan;
        run("single NaN", v);
    }
    {
        auto v = filled(4096);
        v[3]    = kInf;
        v[2000] = -kInf;
        run("infinities, finite limit", v);
        run("infinities, infinite limit", v, kInf);
    }
    {
        auto v = filled(5000);
        v[9]  = 100.0F;
        v[10] = -100.0F;
        v[11] = 50.0F; // exactly the limit below: not offending
        run("limit exceeded", v, 50.0F);
    }
    {
        // More elements than the grid covers in one pass (1024 blocks of 256), so threads loop and
        // the lowest index must win across both threads and blocks.
        auto v = filled(600000);
        v[599999] = kNan;
        v[262147] = 1.0e30F;
        v[262144 + 3] = -kInf; // same thread as element 3
        v[3]          = kNan;
        v[100000]     = kInf;
        run("multiple, lowest index wins", v);
    }
    for (const std::size_t n : {1U, 31U, 255U, 256U, 257U, 262145U, 1000003U}) {
        auto v = filled(n);
        v[n - 1] = kNan;
        run(("tail element, n=" + std::to_string(n)).c_str(), v);
    }
    {
        auto v = filled(1000);
        v[0] = kNan;
        run("first element", v);
    }
    return failures;
}

int run_boundary_cases(cudaStream_t stream) {
    int failures = 0;
    const auto run = [&](const char* label, DType dtype, std::vector<float> input, float limit) {
        Probe probe;
        failures += probe.scan(std::string(name(dtype)) + " " + label, dtype, input, limit, 1, stream);
    };
    // FP16: the largest finite value is not offending at its own limit, one representable step below
    // the limit is not, and a limit under it makes both signs offend.
    run("fp16 max at its limit", DType::FP16, {65504.0F, -65504.0F, 65488.0F, 0.0F}, 65504.0F);
    run("fp16 max over a lower limit", DType::FP16, {65504.0F, -65504.0F, 1.0F}, 65503.0F);
    run("fp16 overflowed value", DType::FP16, {1.0F, 65520.0F, 2.0F}, 65504.0F); // rounds to +inf
    run("fp16 subnormal and zero", DType::FP16, {0.0F, -0.0F, 6.0e-8F, 1.0e-6F}, 65504.0F);
    // BF16 and FP32: finite maxima pass an infinite limit; one ulp over a finite limit offends.
    run("bf16 max with infinite limit", DType::BF16, {3.3895314e38F, -3.3895314e38F}, kInf);
    run("fp32 max with infinite limit", DType::FP32, {std::numeric_limits<float>::max(),
                                                      -std::numeric_limits<float>::max()}, kInf);
    run("fp32 one ulp over the limit", DType::FP32, {1.0F, std::nextafter(1.0F, 2.0F)}, 1.0F);
    run("bf16 one step over the limit", DType::BF16, {1.0F, 1.0078125F}, 1.0F);
    run("fp32 subnormal", DType::FP32, {std::numeric_limits<float>::denorm_min(), 1.0F}, kLimit);
    // NaNs of every sign and payload.
    run("fp32 negative NaN with payload", DType::FP32, {1.0F, std::bit_cast<float>(0xffc12345U)}, kLimit);
    return failures;
}

// Calls from several sites accumulate in one record: the first site to see a bad value owns it,
// later sites only add to the count, and the same site keeps the lowest index.
int run_multi_site_case(cudaStream_t stream) {
    int failures = 0;
    Probe probe;
    std::vector<float> clean(1000, 1.0F);
    std::vector<float> nan_at_60(1000, 1.0F);
    nan_at_60[60] = kNan;
    std::vector<float> inf_at_2(1000, 1.0F);
    inf_at_2[2] = kInf;
    std::vector<float> nan_at_10(1000, 1.0F);
    nan_at_10[10] = kNan;
    failures += probe.scan("multi-site: clean first", DType::BF16, clean, kLimit, 5, stream);
    failures += probe.scan("multi-site: first site to see a bad value", DType::BF16, nan_at_60, kLimit, 5, stream);
    failures += probe.scan("multi-site: later site only counts", DType::FP32, inf_at_2, kLimit, 9, stream);
    failures += probe.scan("multi-site: same site, lower index", DType::FP16, nan_at_10, kLimit, 5, stream);
    failures += probe.scan("multi-site: clean after findings", DType::BF16, clean, kLimit, 9, stream);

    // A cleared record starts over, and site 0 (stored as 1) is a valid id.
    probe.clear();
    failures += probe.check("multi-site: cleared record");
    failures += probe.scan("multi-site: site 0", DType::FP32, inf_at_2, kLimit, 0, stream);
    return failures;
}

// The probe is one kernel node with fixed arguments: a captured graph must see current data on every
// replay, and clearing the record between replays must clear the verdict.
int run_graph_case(cudaStream_t stream) {
    int failures = 0;
    constexpr std::size_t n = 100000;
    std::vector<float> clean(n, 0.5F);
    std::vector<float> bad = clean;
    bad[n - 5] = kNan;
    bad[77]    = kInf;

    const Encoded clean_data = encode(DType::BF16, clean);
    const Encoded bad_data   = encode(DType::BF16, bad);
    GuardedDeviceBuffer values(clean_data.bytes.size());
    values.copy_from_host(clean_data.bytes.data(), clean_data.bytes.size());
    Tensor tensor(values.data(), DType::BF16, {static_cast<std::int32_t>(n)});
    Probe probe;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    ops::numeric_tripwire_scan(tensor, 12, kLimit, probe.tensor(), stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");

    const auto replay = [&](const char* label, const Encoded& data, bool clear_first) {
        if (clear_first) { probe.clear(); }
        values.copy_from_host(data.bytes.data(), data.bytes.size());
        cuda_check(cudaGraphLaunch(executable, stream), "launch");
        cuda_synchronize(stream);
        probe.expected().apply(12, tripwire_oracle_scan(data.widened, kLimit));
        failures += probe.check(label);
    };
    replay("graph replay 1, clean", clean_data, true);
    replay("graph replay 2, bad data", bad_data, true);
    replay("graph replay 3, cleared and clean again", clean_data, true);
    replay("graph replay 4, bad data again", bad_data, true);

    cuda_check(cudaGraphExecDestroy(executable), "destroy exec");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    return failures;
}

// Invalid calls throw before anything is launched and leave the record alone.
int run_rejection_cases(cudaStream_t stream) {
    int failures = 0;
    Probe probe;
    GuardedDeviceBuffer values(64 * 4);
    Tensor ok(values.data(), DType::FP32, {64});

    const auto expect_throw = [&](const char* label, auto&& call) {
        try {
            call();
            std::cerr << label << ": expected an exception\n";
            ++failures;
        } catch (const std::exception&) {}
    };
    GuardedDeviceBuffer ints(64 * 4);
    Tensor wrong_dtype(ints.data(), DType::I32, {64});
    expect_throw("integer values", [&] { ops::numeric_tripwire_scan(wrong_dtype, 1, kLimit, probe.tensor(), stream); });
    Tensor strided(values.data(), DType::FP32, {32, 2});
    strided.nb[1] *= 2; // a column stride wider than the column: not contiguous
    expect_throw("non-contiguous values", [&] { ops::numeric_tripwire_scan(strided, 1, kLimit, probe.tensor(), stream); });
    expect_throw("zero limit", [&] { ops::numeric_tripwire_scan(ok, 1, 0.0F, probe.tensor(), stream); });
    expect_throw("negative limit", [&] { ops::numeric_tripwire_scan(ok, 1, -1.0F, probe.tensor(), stream); });
    expect_throw("NaN limit", [&] { ops::numeric_tripwire_scan(ok, 1, kNan, probe.tensor(), stream); });
    expect_throw("out-of-range site", [&] {
        ops::numeric_tripwire_scan(ok, std::numeric_limits<std::uint32_t>::max(), kLimit, probe.tensor(), stream);
    });
    GuardedDeviceBuffer small_record(8);
    Tensor small(small_record.data(), DType::U8, {8});
    expect_throw("record too small", [&] { ops::numeric_tripwire_scan(ok, 1, kLimit, small, stream); });
    GuardedDeviceBuffer odd_record(32);
    Tensor misaligned(static_cast<std::uint8_t*>(odd_record.data()) + 4, DType::U8, {16});
    expect_throw("misaligned record", [&] { ops::numeric_tripwire_scan(ok, 1, kLimit, misaligned, stream); });
    Tensor wrong_record(odd_record.data(), DType::I32, {8});
    expect_throw("record of the wrong dtype", [&] { ops::numeric_tripwire_scan(ok, 1, kLimit, wrong_record, stream); });
    // A Tensor cannot be built with an empty extent, so "empty tensor" means a hand-forged view.
    Tensor empty = ok;
    empty.ne[0]  = 0;
    expect_throw("empty tensor", [&] { ops::numeric_tripwire_scan(empty, 1, kLimit, probe.tensor(), stream); });
    expect_throw("empty tensor cannot be constructed", [&] { Tensor forbidden(values.data(), DType::FP32, {0}); });

    failures += probe.check("rejections leave the record untouched");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    for (const DType dtype : {DType::FP32, DType::FP16, DType::BF16}) {
        failures += run_dtype_cases(dtype, stream);
    }
    failures += run_boundary_cases(stream);
    failures += run_multi_site_case(stream);
    failures += run_graph_case(stream);
    failures += run_rejection_cases(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " numeric_tripwire_scan public contract\n";
    return failures == 0 ? 0 : 1;
}
