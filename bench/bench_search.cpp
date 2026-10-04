/**
 * @file bench_search.cpp
 * @brief Compares FlatIndex and HnswIndex: build time, queries per second and
 *        recall@10 for several ef values. Build in Release mode.
 *
 * Data: clustered synthetic vectors (points scattered around random centers),
 * which behave much more like real embeddings than uniform random noise.
 * Change the constants below to try other sizes.
 */
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "flat_index.h"
#include "hnsw_index.h"

using namespace vecdb;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kCount = 50'000;   // vectors in the index
constexpr std::size_t kDim = 128;        // dimensions
constexpr std::size_t kClusters = 100;   // centers in the synthetic data
constexpr std::size_t kQueries = 500;    // timed searches
constexpr std::size_t kK = 10;           // results per search

/// Points scattered around random centers.
static std::vector<std::vector<float>> clustered(std::size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    std::mt19937 center_rng(12345);  // same centers for data and queries
    std::vector<std::vector<float>> centers(kClusters, std::vector<float>(kDim));
    for (auto& c : centers)
        for (float& x : c) x = center(center_rng);
    std::vector<std::vector<float>> out(count, std::vector<float>(kDim));
    for (auto& v : out) {
        const auto& c = centers[rng() % kClusters];
        for (std::size_t d = 0; d < kDim; ++d) v[d] = c[d] + noise(rng);
    }
    return out;
}

/// Seconds elapsed since `start`.
static double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int main() {
    std::printf("Kernel: %s | %zu vectors x %zu dims | %zu queries | k = %zu\n\n",
                isa_name(active_isa()), kCount, kDim, kQueries, kK);
    auto data = clustered(kCount, 1);
    auto queries = clustered(kQueries, 2);

    // Build both indexes.
    FlatIndex flat(kDim, Metric::L2);
    auto start = Clock::now();
    for (std::size_t i = 0; i < kCount; ++i) flat.add(i, data[i]);
    std::printf("Flat build: %.2f s\n", seconds_since(start));

    HnswIndex hnsw(kDim, Metric::L2);  // M = 16, ef_construction = 200
    start = Clock::now();
    for (std::size_t i = 0; i < kCount; ++i) hnsw.add(i, data[i]);
    std::printf("HNSW build: %.2f s (max level %d)\n\n", seconds_since(start), hnsw.max_level());

    // Ground truth from the exact index, timed as the Flat baseline.
    std::vector<std::set<std::uint64_t>> truth(kQueries);
    start = Clock::now();
    for (std::size_t q = 0; q < kQueries; ++q)
        for (const auto& r : flat.search(queries[q], kK)) truth[q].insert(r.id);
    const double flat_qps = double(kQueries) / seconds_since(start);

    std::printf("%-8s %-6s %12s %10s %10s\n", "index", "ef", "queries/s", "recall@10", "speedup");
    std::printf("%-8s %-6s %12.0f %10.3f %9.1fx\n", "flat", "-", flat_qps, 1.0, 1.0);

    for (std::size_t ef : {10, 20, 40, 80, 160, 320}) {
        std::size_t hits = 0;
        start = Clock::now();
        for (std::size_t q = 0; q < kQueries; ++q)
            for (const auto& r : hnsw.search(queries[q], kK, ef)) hits += truth[q].count(r.id);
        const double qps = double(kQueries) / seconds_since(start);
        const double recall = double(hits) / double(kQueries * kK);
        std::printf("%-8s %-6zu %12.0f %10.3f %9.1fx\n", "hnsw", ef, qps, recall, qps / flat_qps);
    }
}