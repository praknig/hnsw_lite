/**
 * @file bench_distance.cpp
 * @brief Measures nanoseconds per distance call for every supported version.
 *
 * Build in Release mode, otherwise the numbers are meaningless.
 * Uses a small set of vectors that fits in the CPU cache, so it measures
 * the speed of the math itself, not the speed of main memory.
 */
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "distance.h"
#include "vector_store.h"

using namespace vecdb;

/// Average nanoseconds per call of `fn` over all pairs in `store`.
static double time_kernel(DistanceFn fn, const VectorStore& store, int rounds) {
    const std::size_t count = store.size(), n = store.stride();
    volatile float sink = 0.0f;  // stops the compiler from deleting the work
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; ++r)
        for (NodeId i = 0; i < count; ++i)
            sink = sink + fn(store.get_padded(0).data(), store.get_padded(i).data(), n);
    auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start);
    return ns.count() / (double(rounds) * double(count));
}

int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::printf("Active version: %s\n\n", isa_name(active_isa()));
    std::printf("%-6s %-8s %12s %10s\n", "dim", "version", "ns/call", "speedup");

    for (std::size_t dim : {128, 768, 1536}) {
        VectorStore store(dim);
        std::vector<float> v(dim);
        for (int i = 0; i < 64; ++i) {
            for (float& x : v) x = dist(rng);
            store.add(v);
        }
        const int rounds = int(4'000'000 / (64 * dim)) + 1;

        double scalar_ns = 0.0;
        for (Isa isa : {Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512}) {
            DistanceFn fn = get_distance(Metric::L2, isa);
            if (!fn) continue;
            time_kernel(fn, store, rounds);  // warm-up
            double ns = time_kernel(fn, store, rounds * 4);
            if (isa == Isa::Scalar) scalar_ns = ns;
            std::printf("%-6zu %-8s %12.1f %9.1fx\n", dim, isa_name(isa), ns, scalar_ns / ns);
        }
        std::printf("\n");
    }
}