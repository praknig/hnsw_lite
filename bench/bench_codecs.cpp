/**
 * @file bench_codecs.cpp
 * @brief Times code distances (SQ8, binary) for every instruction set this CPU
 *        supports, next to full-precision float distances.
 *
 * Usage: bench_codecs [dimension, default 768]. Run on a quiet machine: the
 * numbers are nanoseconds per distance, best of several rounds.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "distance.h"

using namespace vecdb;

namespace {

template <class F>
double ns_per_call(std::size_t calls, F&& f) {
    double best = 1e30;
    for (int round = 0; round < 5; ++round) {
        const auto start = std::chrono::steady_clock::now();
        f();
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        best = std::min(best, ns / double(calls));
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t dim = argc > 1 ? std::size_t(std::strtoul(argv[1], nullptr, 10)) : 768;
    const std::size_t n = 20000;  // stored vectors (codes), visited in order
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<float> table(dim), step(dim), floats(n * dim), query(dim);
    for (float& x : table) x = u(rng);
    for (float& x : step) x = std::abs(u(rng)) / 255;
    for (float& x : floats) x = u(rng);
    for (float& x : query) x = u(rng);
    std::vector<std::uint8_t> sq8(n * dim), bits(n * ((dim + 7) / 8));
    for (auto& b : sq8) b = std::uint8_t(byte(rng));
    for (auto& b : bits) b = std::uint8_t(byte(rng));
    std::printf("dimension %zu, %zu vectors; nanoseconds per distance (best of 5)\n\n", dim, n);
    std::printf("%-8s %12s %12s %12s %12s\n", "kernels", "float L2", "SQ8 L2", "SQ8 dot", "binary");
    volatile float sink = 0;
    for (Isa isa : {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon}) {
        if (!isa_supported(isa)) continue;
        const DistanceFn l2 = get_distance(Metric::L2, isa);
        const CodeKernels k = get_code_kernels(isa);
        const double f = ns_per_call(n, [&] { for (std::size_t i = 0; i < n; ++i) sink = sink + l2(query.data(), &floats[i * dim], dim); });
        const double a = ns_per_call(n, [&] { for (std::size_t i = 0; i < n; ++i) sink = sink + k.sq8_l2(table.data(), step.data(), &sq8[i * dim], dim); });
        const double b = ns_per_call(n, [&] { for (std::size_t i = 0; i < n; ++i) sink = sink + k.sq8_dot(table.data(), &sq8[i * dim], dim); });
        const std::size_t cs = (dim + 7) / 8;
        const double c = ns_per_call(n, [&] { for (std::size_t i = 0; i < n; ++i) sink = sink + k.bits_sum(table.data(), &bits[i * cs], dim); });
        std::printf("%-8s %12.1f %12.1f %12.1f %12.1f\n", isa_name(isa), f, a, b, c);
    }
    std::printf("\nMemory per vector: floats %zu bytes, SQ8 %zu bytes, binary %zu bytes\n", dim * 4, dim, (dim + 7) / 8);
    return 0;
}