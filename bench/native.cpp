/**
 * @file native_compare.cpp
 * @brief hnsw-lite against hnswlib in C++, on identical data, without Python: for
 * measuring search optimizations before and after, and for profiling.
 *
 * Build (hnswlib is header-only; fetch its headers once):
 *
 *   mkdir -p build/hnswlib && cd build/hnswlib
 *   for f in hnswlib.h hnswalg.h space_l2.h space_ip.h visited_list_pool.h bruteforce.h stop_condition.h; do
 *     curl -sfLO https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/hnswlib/$f; done
 *   cd ../..
 *   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target hnsw_lite
 *   g++ -std=c++20 -O3 -march=native -Iinclude -Isrc -Ibuild/hnswlib bench/native_compare.cpp \
 *       -Lbuild -lhnsw_lite -pthread -o build/native_compare
 *
 * Run:
 *
 *   build/native_compare                 both libraries on the same data: time per query and agreement
 *   build/native_compare save            builds hnsw-lite's index and queries into files (for the modes below)
 *   build/native_compare time            hnsw-lite on the saved index and queries: time per query
 *   build/native_compare profile         the same, searching repeatedly: run under perf record
 *
 * To measure a change: build this program before it (keep the binary: the library is
 * linked in statically), run `save` once, then alternate `time` between the old and
 * new binaries. Their `profile` output (a checksum of results) must be identical.
 *
 * Options (environment): N (20000), DIM (128), QUERIES (2000), EF (64), DIR (/tmp).
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "hnsw_index.h"
#include "hnswlib.h"

namespace {

std::size_t env(const char* name, std::size_t fallback) {
    const char* v = std::getenv(name);
    return v ? std::stoul(v) : fallback;
}

std::string dir() {
    const char* v = std::getenv("DIR");
    return v ? v : "/tmp";
}

/// The best time per query over `rounds` passes, in microseconds.
template <class F>
double time_per_query(std::size_t queries, int rounds, F&& one) {
    double best = 1e300;
    for (int r = 0; r < rounds; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t q = 0; q < queries; ++q) one(q);
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / double(queries));
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = env("N", 20000), dim = env("DIM", 128), nq = env("QUERIES", 2000), ef = env("EF", 64), k = 10;
    const std::string mode = argc > 1 ? argv[1] : "compare";
    const std::string index_path = dir() + "/native_compare.idx", queries_path = dir() + "/native_compare.queries";

    if (mode == "time" || mode == "profile") {
        std::unique_ptr<vecdb::HnswIndex> index;
        std::vector<float> q(nq * dim);
        std::FILE* f = std::fopen(queries_path.c_str(), "rb");
        const bool read = f && std::fread(q.data(), sizeof(float), q.size(), f) == q.size();
        if (f) std::fclose(f);
        try {
            if (read) index = vecdb::HnswIndex::load(index_path);
        } catch (const std::exception&) {
        }
        if (!index) {
            std::fprintf(stderr, "run `native_compare save` first (with the same N, DIM and QUERIES)\n");
            return 1;
        }
        if (mode == "time") {
            const double t = time_per_query(nq, 7, [&](std::size_t i) { (void)index->search({&q[i * dim], dim}, k, ef); });
            std::printf("hnsw-lite: %.1f us/query\n", t);
        } else {
            std::size_t checksum = 0;
            for (int r = 0; r < 60; ++r)
                for (std::size_t i = 0; i < nq; ++i) checksum += index->search({&q[i * dim], dim}, k, ef)[0].id;
            std::printf("checksum of top results: %zu\n", checksum);
        }
        return 0;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<float> data(n * dim), queries(nq * dim);
    for (float& x : data) x = u(rng);
    for (float& x : queries) x = u(rng);
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;
    vecdb::HnswIndex lite(dim, vecdb::Metric::L2);  // M 16, ef_construction 200: hnswlib's settings below
    lite.add_batch(ids, data, 1);

    if (mode == "save") {
        lite.save(index_path);
        std::FILE* f = std::fopen(queries_path.c_str(), "wb");
        std::fwrite(queries.data(), sizeof(float), queries.size(), f);
        std::fclose(f);
        std::printf("saved %s and %s\n", index_path.c_str(), queries_path.c_str());
        return 0;
    }

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> other(&space, n, 16, 200, 42);
    for (std::size_t i = 0; i < n; ++i) other.addPoint(&data[i * dim], i);
    other.setEf(ef);
    const double t_lite = time_per_query(nq, 5, [&](std::size_t q) { (void)lite.search({&queries[q * dim], dim}, k, ef); });
    const double t_other = time_per_query(nq, 5, [&](std::size_t q) { auto r = other.searchKnn(&queries[q * dim], k); (void)r; });
    double agreement = 0;
    const std::size_t checked = std::min<std::size_t>(nq, 200);
    for (std::size_t q = 0; q < checked; ++q) {
        std::set<std::uint64_t> a;
        for (const auto& r : lite.search({&queries[q * dim], dim}, k, ef)) a.insert(r.id);
        auto pq = other.searchKnn(&queries[q * dim], k);
        std::size_t both = 0;
        for (; !pq.empty(); pq.pop()) both += a.count(pq.top().second);
        agreement += double(both) / double(k);
    }
    std::printf("%zu vectors of %zu dimensions, ef %zu: hnsw-lite %.1f us/query, hnswlib %.1f us/query; their top %zu agree %.0f%%\n",
                n, dim, ef, t_lite, t_other, k, 100 * agreement / double(checked));
    return 0;
}