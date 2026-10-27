/**
 * @file bench_locality.cpp
 * @brief Whether memory stalls dominate HNSW search on this machine: nanoseconds per
 *        distance computation with an index that fits in the CPU's cache, and with one
 *        several times larger. A large gap means a cache-friendlier node layout (vector
 *        and links stored together) or prefetching would pay off; a small one, that it
 *        would not. Results are the same either way: only time is measured.
 *
 * Usage: bench_locality [large vectors, default 400000] [dimension, default 128]
 * The large index takes 400000 * 128 * 4 bytes = 205 MB of vectors, beyond most desktop
 * caches (8 to 64 MB); building it takes a few minutes. Run on a quiet machine.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "hnsw_index.h"

using namespace vecdb;

namespace {
struct Result {
    double us_per_search, ns_per_distance, distances_per_search;
};

Result measure(std::size_t n, std::size_t dim) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> data(n * dim);
    for (float& x : data) x = nd(rng);
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;
    HnswIndex h(dim, Metric::L2);
    h.add_batch(ids, data, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::vector<float>> queries(500, std::vector<float>(dim));
    for (auto& q : queries) for (float& x : q) x = nd(rng);
    Result best{1e18, 1e18, 0};
    for (int round = 0; round < 5; ++round) {
        std::size_t distances = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& q : queries) {
            SearchStats st;
            SearchOptions o;
            o.stats = &st;
            (void)h.search(q, 10, 128, o);
            distances += st.distance_computations;
        }
        const double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        if (ns / 1000.0 / double(queries.size()) < best.us_per_search)
            best = {ns / 1000.0 / double(queries.size()), ns / double(distances), double(distances) / double(queries.size())};
    }
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    const std::size_t large = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 400000;
    const std::size_t dim = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 128;
    const std::size_t small = 10000;  // 10000 * 128 * 4 = 5 MB: in cache on nearly any CPU
    std::printf("HNSW search, dimension %zu, k 10, ef 128 (best of 5 rounds of 500 queries)\n\n", dim);
    std::printf("%10s %12s %16s %20s\n", "vectors", "MB vectors", "us per search", "ns per distance");
    const Result a = measure(small, dim);
    std::printf("%10zu %12.0f %16.1f %20.1f\n", small, double(small * dim * 4) / 1e6, a.us_per_search, a.ns_per_distance);
    std::fflush(stdout);
    const Result b = measure(large, dim);
    std::printf("%10zu %12.0f %16.1f %20.1f\n", large, double(large * dim * 4) / 1e6, b.us_per_search, b.ns_per_distance);
    std::printf("\nEach distance costs %.2fx as much in the large index.\n", b.ns_per_distance / a.ns_per_distance);
    std::printf("Above about 1.5x, memory stalls dominate: a cache-friendlier layout or prefetching would help.\n");
    std::printf("Near 1x, the data stays in cache (or the CPU hides the latency): they would not.\n");
}
