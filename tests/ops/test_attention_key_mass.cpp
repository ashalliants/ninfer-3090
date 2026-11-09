// attention_key_mass against an FP64 oracle: per key, the softmax probability mass summed over
// every query row and query head. Cases follow the lossy-prefill scout at the 27B filter layer
// (head_dim 256, 24 query heads over 4 KV heads, 64 query rows) at key counts that end on a
// partial tile (17, 100,003) and on a whole one (4,096), in BF16 and FP16, plus a small geometry
// whose query group (6 rows x 3 heads = 18) fills no tile. A timing line reports N = 100,000.

#include "ninfer/ops/attention_key_mass.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Each output is a sum of probabilities from FP32 logits of a 256-term dot product; FP32 against
// the FP64 oracle lands near 1e-5 relative. 1e-3 is the brief's bound and leaves the slack for the
// logit error that grows with |scale * q.k|.
constexpr PointwiseCriterion key_mass_criterion() { return {/*absolute*/ 0.0, /*relative*/ 1.0e-3}; }
constexpr double kTotalMassRelative = 1.0e-3;

struct Shape {
    std::int32_t head_dim;
    std::int32_t q_heads;
    std::int32_t rows;
    std::int32_t kv_heads;
    std::int32_t keys;
};

std::uint16_t encode(float value, DType dtype) {
    if (dtype == DType::BF16) return f32_to_bf16(value);
    const __half h = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float decode(std::uint16_t bits, DType dtype) {
    if (dtype == DType::BF16) return bf16_to_f32(bits);
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
}

// q [D, Hq, R] and k [D, Hkv, N], already rounded to the stored dtype. Each KV group has a shared
// direction u_g; every query of the group leans on it and a sparse set of keys carries it with
// varying strength, so the softmax has real peaks (up to ~e^8 above the floor) instead of being
// flat, and the planted keys collect most of the mass.
void make_inputs(const Shape& s, DType dtype, std::uint32_t seed, std::vector<std::uint16_t>& q_bits,
                 std::vector<std::uint16_t>& k_bits, std::vector<float>& q, std::vector<float>& k) {
    const std::size_t d = static_cast<std::size_t>(s.head_dim);
    std::vector<float> direction(d * static_cast<std::size_t>(s.kv_heads));
    fill_uniform(direction, seed, -1.0F, 1.0F);
    for (auto& value : direction) value = value < 0.0F ? -1.0F : 1.0F;

    q.resize(d * static_cast<std::size_t>(s.q_heads) * static_cast<std::size_t>(s.rows));
    k.resize(d * static_cast<std::size_t>(s.kv_heads) * static_cast<std::size_t>(s.keys));
    fill_uniform(q, seed + 1, -1.0F, 1.0F);
    fill_uniform(k, seed + 2, -1.0F, 1.0F);
    const std::int32_t group = s.q_heads / s.kv_heads;
    for (std::int32_t r = 0; r < s.rows; ++r) {
        for (std::int32_t h = 0; h < s.q_heads; ++h) {
            const float* u = direction.data() + d * static_cast<std::size_t>(h / group);
            float* v = q.data() + d * (static_cast<std::size_t>(h) + static_cast<std::size_t>(s.q_heads) * r);
            for (std::size_t i = 0; i < d; ++i) v[i] = 0.5F * v[i] + 0.5F * u[i];
        }
    }
    for (std::int32_t j = 0; j < s.keys; ++j) {
        const std::int32_t pick = j % 97;
        const float strength    = pick == 13 ? 1.0F : (pick == 41 ? 0.5F : (pick == 70 ? 0.25F : 0.0F));
        if (strength == 0.0F) continue;
        for (std::int32_t g = 0; g < s.kv_heads; ++g) {
            const float* u = direction.data() + d * static_cast<std::size_t>(g);
            float* v = k.data() + d * (static_cast<std::size_t>(g) + static_cast<std::size_t>(s.kv_heads) * j);
            for (std::size_t i = 0; i < d; ++i) v[i] = 0.5F * v[i] + strength * u[i];
        }
    }
    q_bits.resize(q.size());
    k_bits.resize(k.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q_bits[i] = encode(q[i], dtype);
        q[i]      = decode(q_bits[i], dtype);
    }
    for (std::size_t i = 0; i < k.size(); ++i) {
        k_bits[i] = encode(k[i], dtype);
        k[i]      = decode(k_bits[i], dtype);
    }
}

// FP64: for every (row, head), logits over all keys, a max-shifted softmax, and its probabilities
// added to that worker's own accumulator; the workers' accumulators are summed in a fixed order.
std::vector<double> key_mass_oracle(const Shape& s, const std::vector<float>& q, const std::vector<float>& k,
                                    double scale) {
    const std::size_t d      = static_cast<std::size_t>(s.head_dim);
    const std::size_t n      = static_cast<std::size_t>(s.keys);
    const std::int32_t group = s.q_heads / s.kv_heads;
    const std::int32_t tasks = s.q_heads * s.rows;
    const unsigned workers   = std::max(1U, std::min(std::thread::hardware_concurrency(), 32U));
    std::vector<std::vector<double>> partial(workers, std::vector<double>(n, 0.0));
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            std::vector<double> logits(n);
            std::vector<double>& acc = partial[w];
            for (std::int32_t task = static_cast<std::int32_t>(w); task < tasks;
                 task += static_cast<std::int32_t>(workers)) {
                const std::int32_t h = task % s.q_heads;
                const std::int32_t r = task / s.q_heads;
                const std::int32_t g = h / group;
                const float* qv = q.data() + d * (static_cast<std::size_t>(h) + static_cast<std::size_t>(s.q_heads) * r);
                double maximum = -INFINITY;
                for (std::size_t j = 0; j < n; ++j) {
                    const float* kv = k.data() + d * (static_cast<std::size_t>(g) + static_cast<std::size_t>(s.kv_heads) * j);
                    double dot = 0.0;
                    for (std::size_t i = 0; i < d; ++i) dot += static_cast<double>(qv[i]) * static_cast<double>(kv[i]);
                    logits[j] = scale * dot;
                    maximum   = std::max(maximum, logits[j]);
                }
                double sum = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    logits[j] = std::exp(logits[j] - maximum);
                    sum += logits[j];
                }
                for (std::size_t j = 0; j < n; ++j) acc[j] += logits[j] / sum;
            }
        });
    }
    for (auto& t : pool) t.join();
    std::vector<double> out(n, 0.0);
    for (const auto& p : partial) {
        for (std::size_t j = 0; j < n; ++j) out[j] += p[j];
    }
    return out;
}

const char* dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "f16"; }

int run_case(const Shape& s, DType dtype, std::uint32_t seed) {
    const std::string label = std::string("attention_key_mass ") + dtype_name(dtype) + " q[" +
                              std::to_string(s.head_dim) + "," + std::to_string(s.q_heads) + "," +
                              std::to_string(s.rows) + "] k[" + std::to_string(s.head_dim) + "," +
                              std::to_string(s.kv_heads) + "," + std::to_string(s.keys) + "]";
    std::vector<std::uint16_t> q_bits, k_bits;
    std::vector<float> q, k;
    make_inputs(s, dtype, seed, q_bits, k_bits, q, k);
    const double scale = 1.0 / std::sqrt(static_cast<double>(s.head_dim));
    const std::vector<double> reference = key_mass_oracle(s, q, k, scale);

    DeviceBuffer dq = to_device(q_bits);
    DeviceBuffer dk = to_device(k_bits);
    GuardedDeviceBuffer out(static_cast<std::size_t>(s.keys) * sizeof(float));
    out.fill(0xff); // NaN: every element must be overwritten
    DeviceArena work(ops::attention_key_mass_workspace_capacity_bytes(s.q_heads, s.rows, s.keys));

    Tensor tq(dq.p, dtype, {s.head_dim, s.q_heads, s.rows});
    Tensor tk(dk.p, dtype, {s.head_dim, s.kv_heads, s.keys});
    Tensor tout(out.data(), DType::FP32, {s.keys});
    ops::attention_key_mass(tq, tk, static_cast<float>(scale), tout, work, nullptr);
    cuda_synchronize();

    const std::vector<float> got_f = from_device<float>(out.data(), static_cast<std::size_t>(s.keys));
    const std::vector<double> got(got_f.begin(), got_f.end());
    int failures = verify_pointwise(label, got, reference, key_mass_criterion());
    failures += out.verify_guards(label);

    double total = 0.0;
    for (const double v : got) total += v;
    const double expected_total = static_cast<double>(s.rows) * s.q_heads;
    if (!(std::abs(total - expected_total) <= kTotalMassRelative * expected_total)) {
        std::cerr << label << ": total mass " << total << " expected " << expected_total << '\n';
        ++failures;
    }
    std::cout << (failures ? "FAIL " : "ok   ") << label << " total=" << total << '\n';
    return failures;
}

// The brief's bound for the 27B scout at 100K keys is < 20 ms. Printed, not asserted: timing on a
// shared card is evidence, not a correctness verdict.
void report_timing() {
    const Shape s{256, 24, 64, 4, 100000};
    std::vector<std::uint16_t> q_bits, k_bits;
    std::vector<float> q, k;
    make_inputs(s, DType::BF16, 5501U, q_bits, k_bits, q, k);
    DeviceBuffer dq = to_device(q_bits);
    DeviceBuffer dk = to_device(k_bits);
    DeviceBuffer dout(static_cast<std::size_t>(s.keys) * sizeof(float));
    DeviceArena work(ops::attention_key_mass_workspace_capacity_bytes(s.q_heads, s.rows, s.keys));
    Tensor tq(dq.p, DType::BF16, {s.head_dim, s.q_heads, s.rows});
    Tensor tk(dk.p, DType::BF16, {s.head_dim, s.kv_heads, s.keys});
    Tensor tout(dout.p, DType::FP32, {s.keys});
    const float scale = 1.0F / 16.0F;

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
    for (int i = 0; i < 3; ++i) ops::attention_key_mass(tq, tk, scale, tout, work, stream);
    constexpr int kIterations = 20;
    cuda_check(cudaEventRecord(start, stream), "cudaEventRecord");
    for (int i = 0; i < kIterations; ++i) ops::attention_key_mass(tq, tk, scale, tout, work, stream);
    cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    std::printf("TIMING attention_key_mass bf16 q[256,24,64] k[256,4,100000]: %.3f ms/call "
                "(mean of %d; brief expects < 20 ms)\n",
                ms / kIterations, kIterations);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

template <typename Body>
int expect_invalid(const char* label, Body&& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << label << ": expected std::invalid_argument\n";
    return 1;
}

int run_validation() {
    DeviceBuffer a(256 * 24 * 64 * 2), b(256 * 4 * 64 * 2), c(64 * 4);
    DeviceArena work(ops::attention_key_mass_workspace_capacity_bytes(24, 64, 64));
    int failures = 0;
    failures += expect_invalid("mixed q/k dtypes", [&] {
        Tensor q(a.p, DType::FP16, {256, 24, 64}), k(b.p, DType::BF16, {256, 4, 64});
        Tensor out(c.p, DType::FP32, {64});
        ops::attention_key_mass(q, k, 0.0625F, out, work, nullptr);
    });
    failures += expect_invalid("q_heads not a multiple of kv_heads", [&] {
        Tensor q(a.p, DType::BF16, {256, 6, 64}), k(b.p, DType::BF16, {256, 4, 64});
        Tensor out(c.p, DType::FP32, {64});
        ops::attention_key_mass(q, k, 0.0625F, out, work, nullptr);
    });
    failures += expect_invalid("out length differs from keys", [&] {
        Tensor q(a.p, DType::BF16, {256, 24, 64}), k(b.p, DType::BF16, {256, 4, 64});
        Tensor out(c.p, DType::FP32, {63});
        ops::attention_key_mass(q, k, 0.0625F, out, work, nullptr);
    });
    failures += expect_invalid("head_dim mismatch", [&] {
        Tensor q(a.p, DType::BF16, {256, 24, 64}), k(b.p, DType::BF16, {128, 4, 64});
        Tensor out(c.p, DType::FP32, {64});
        ops::attention_key_mass(q, k, 0.0625F, out, work, nullptr);
    });
    failures += expect_invalid("FP32 out required", [&] {
        Tensor q(a.p, DType::BF16, {256, 24, 64}), k(b.p, DType::BF16, {256, 4, 64});
        Tensor out(c.p, DType::BF16, {64});
        ops::attention_key_mass(q, k, 0.0625F, out, work, nullptr);
    });
    failures += expect_invalid("capacity query rejects zero keys",
                               [] { (void)ops::attention_key_mass_workspace_capacity_bytes(24, 64, 0); });
    std::cout << (failures ? "FAIL" : "ok  ") << " attention_key_mass validation\n";
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_validation();
    failures += run_case({256, 24, 64, 4, 17}, DType::BF16, 5101U);
    failures += run_case({256, 24, 64, 4, 4096}, DType::BF16, 5102U);
    failures += run_case({256, 24, 64, 4, 100003}, DType::BF16, 5103U);
    failures += run_case({256, 24, 64, 4, 4096}, DType::FP16, 5104U);
    failures += run_case({128, 3, 6, 1, 129}, DType::FP16, 5105U);
    failures += run_case({128, 6, 6, 2, 1}, DType::BF16, 5106U);
    report_timing();
    std::cout << (failures ? "FAIL" : "OK") << " attention_key_mass\n";
    return failures ? 1 : 0;
}
