/**
 * @file bench_baseline.cpp
 * @brief Baseline benchmark for performance work: build time, memory, and for
 *        several ef values, queries per second, latency percentiles and recall.
 *
 * Add --threads=N anywhere to build with add_batch on N threads (0: all cores)
 * instead of adding vectors one by one.
 *
 * Usage, generated data:
 *   bench_baseline [n] [dim] [l2|cosine] [queries] [M] [ef_construction]
 *   Defaults: 200000 vectors, 128 dimensions, l2, 1000 queries, M 16, ef_construction 200.
 *
 * Usage, real data in the standard .fvecs / .ivecs formats (for example SIFT1M:
 * sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs):
 *   bench_baseline --files <base.fvecs> <query.fvecs> <truth.ivecs> [l2|cosine] [queries] [M] [ef_construction]
 *   The ground-truth file must list the exact nearest neighbors of each query,
 *   computed with the same metric.
 *
 * Data: 100 clusters in a 16-dimensional latent space, mapped into `dim`
 * dimensions with a little noise (like real embeddings), generated deterministically.
 * Each ef value is measured over 5 passes after a warm-up pass; queries per
 * second come from the median pass, which damps timing jitter. Single-threaded.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "flat_index.h"
#include "hnsw_index.h"

using namespace vecdb;
using Clock = std::chrono::steady_clock;

namespace {

/// Resident memory of this process in MiB (Linux only; -1 elsewhere).
double rss_mib() {
#ifdef __linux__
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, "VmRSS:", 6) == 0) {
                std::fclose(f);
                return std::atof(line + 6) / 1024.0;
            }
        std::fclose(f);
    }
#endif
    return -1.0;
}

/// Realistic data: clustered points in a 16-dimensional latent space, mapped
/// into `dim` dimensions by a fixed random linear map, plus a little noise. Real
/// embeddings vary along far fewer directions than their dimension; plain
/// isotropic noise in every dimension makes nearest neighbors meaningless.
std::vector<float> clustered(std::size_t n, std::size_t dim, unsigned seed) {
    const std::size_t latent = 16, clusters = 100;
    std::mt19937 map_rng(12345);  // the same map for data and queries
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> map(dim * latent), centers(clusters * latent);
    for (float& x : map) x = unit(map_rng) / 4.0f;  // 1 / sqrt(latent)
    for (float& x : centers) x = unit(map_rng);
    std::mt19937 rng(seed);
    std::normal_distribution<float> spread(0.0f, 0.5f), noise(0.0f, 0.02f);
    std::vector<float> out(n * dim), z(latent);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = rng() % clusters;
        for (std::size_t l = 0; l < latent; ++l) z[l] = centers[c * latent + l] + spread(rng);
        for (std::size_t d = 0; d < dim; ++d) {
            float s = noise(rng);
            for (std::size_t l = 0; l < latent; ++l) s += map[d * latent + l] * z[l];
            out[i * dim + d] = s;
        }
    }
    return out;
}
/// Reads an .fvecs file: each vector is a 4-byte dimension, then that many floats.
std::vector<float> read_fvecs(const char* path, std::size_t& dim, std::size_t limit) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::vector<float> out;
    std::int32_t d = 0;
    std::size_t count = 0;
    while (count < limit && std::fread(&d, 4, 1, f) == 1) {
        if (d <= 0) { std::fprintf(stderr, "bad dimension in %s\n", path); std::exit(1); }
        dim = std::size_t(d);
        const std::size_t at = out.size();
        out.resize(at + dim);
        if (std::fread(&out[at], 4, dim, f) != dim) { std::fprintf(stderr, "truncated %s\n", path); std::exit(1); }
        ++count;
    }
    std::fclose(f);
    return out;
}

/// Reads an .ivecs file of nearest-neighbor IDs (same layout, 4-byte integers).
std::vector<std::vector<std::int32_t>> read_ivecs(const char* path, std::size_t limit) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::vector<std::vector<std::int32_t>> out;
    std::int32_t d = 0;
    while (out.size() < limit && std::fread(&d, 4, 1, f) == 1) {
        std::vector<std::int32_t> row(std::size_t(d > 0 ? d : 0));
        if (std::fread(row.data(), 4, row.size(), f) != row.size()) { std::fprintf(stderr, "truncated %s\n", path); std::exit(1); }
        out.push_back(std::move(row));
    }
    std::fclose(f);
    return out;
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, std::size_t(p * double(v.size())))];
}

}  // namespace

int main(int argc_in, char** argv_in) {
    // --threads=N (anywhere): build with add_batch on N threads (0: all cores);
    // without it, vectors are added one by one.
    std::size_t threads = 1;
    std::vector<char*> kept;
    for (int i = 0; i < argc_in; ++i) {
        const std::string a = argv_in[i];
        if (a.rfind("--threads=", 0) == 0) {
            threads = std::strtoull(a.c_str() + 10, nullptr, 10);
        } else {
            kept.push_back(argv_in[i]);
        }
    }
    const int argc = static_cast<int>(kept.size());
    char** argv = kept.data();
    const bool files = argc > 1 && std::string(argv[1]) == "--files";
    if (files && argc < 5) {
        std::fprintf(stderr, "usage: bench_baseline --files <base.fvecs> <query.fvecs> <truth.ivecs> [l2|cosine] [queries] [M] [ef_construction]\n");
        return 1;
    }
    const int a = files ? 4 : 0;  // offset of the shared arguments
    const Metric metric = (argc > a + 3 && std::string(argv[a + 3]) == "cosine") ? Metric::Cosine : Metric::L2;
    std::size_t nq = argc > a + 4 ? std::strtoull(argv[a + 4], nullptr, 10) : 1000;
    HnswParams params;
    if (argc > a + 5) params.M = std::strtoull(argv[a + 5], nullptr, 10);
    if (argc > a + 6) params.ef_construction = std::strtoull(argv[a + 6], nullptr, 10);
    const std::size_t k = 10;

    std::size_t n = 0, dim = 0;
    std::vector<float> data, queries;
    std::vector<std::set<std::uint64_t>> truth;
    if (files) {
        data = read_fvecs(argv[2], dim, SIZE_MAX);
        n = data.size() / dim;
        std::size_t qdim = 0;
        queries = read_fvecs(argv[3], qdim, nq);
        if (qdim != dim) { std::fprintf(stderr, "query dimension %zu differs from base dimension %zu\n", qdim, dim); return 1; }
        nq = queries.size() / dim;
        const auto gt = read_ivecs(argv[4], nq);
        if (gt.size() < nq) { std::fprintf(stderr, "ground truth has only %zu rows\n", gt.size()); return 1; }
        truth.resize(nq);
        for (std::size_t q = 0; q < nq; ++q)
            for (std::size_t j = 0; j < std::min(k, gt[q].size()); ++j) truth[q].insert(std::uint64_t(gt[q][j]));
    } else {
        n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200000;
        dim = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 128;
        data = clustered(n, dim, 1);
        queries = clustered(nq, dim, 2);
    }
    std::printf("hnsw-lite baseline | kernel: %s | %zu x %zu %s | %zu queries | M %zu, ef_construction %zu | build: %s\n",
                isa_name(active_isa()), n, dim, metric == Metric::L2 ? "l2" : "cosine", nq, params.M,
                params.ef_construction,
                threads == 1 ? "one by one" : ("add_batch, " + (threads == 0 ? std::string("all cores") : std::to_string(threads) + " threads")).c_str());

    if (!files) {  // exact answers for recall by brute force (not timed)
        FlatIndex flat(dim, metric);
        for (std::size_t i = 0; i < n; ++i) flat.add(i, std::span<const float>(&data[i * dim], dim));
        truth.resize(nq);
        for (std::size_t q = 0; q < nq; ++q)
            for (const auto& r : flat.search(std::span<const float>(&queries[q * dim], dim), k)) truth[q].insert(r.id);
    }

    const double rss_before = rss_mib();
    HnswIndex index(dim, metric, params);
    const auto t0 = Clock::now();
    if (threads == 1) {
        for (std::size_t i = 0; i < n; ++i) index.add(i, std::span<const float>(&data[i * dim], dim));
    } else {
        std::vector<std::uint64_t> ids(n);
        std::iota(ids.begin(), ids.end(), std::uint64_t(0));
        index.add_batch(ids, data, threads);
    }
    const double build_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const double rss_after = rss_mib();
    std::printf("build: %.2f s (%.0f vectors/s) | index memory: %.1f MiB\n\n", build_s, double(n) / build_s,
                rss_after - rss_before);

    std::printf("%6s %12s %10s %10s %10s\n", "ef", "queries/s", "p50 us", "p99 us", "recall@10");
    for (std::size_t ef : std::initializer_list<std::size_t>{16, 32, 64, 128, 256}) {
        std::vector<double> pass_seconds, latencies;
        double recall_sum = 0.0;
        for (int pass = 0; pass <= 5; ++pass) {  // pass 0 warms caches and is not counted
            const auto p0 = Clock::now();
            for (std::size_t q = 0; q < nq; ++q) {
                const auto q0 = Clock::now();
                const auto r = index.search(std::span<const float>(&queries[q * dim], dim), k, ef);
                const double us = std::chrono::duration<double, std::micro>(Clock::now() - q0).count();
                if (pass == 0) {
                    std::size_t hits = 0;
                    for (const auto& x : r) hits += truth[q].count(x.id);
                    recall_sum += double(hits) / double(k);
                } else {
                    latencies.push_back(us);
                }
            }
            if (pass > 0) pass_seconds.push_back(std::chrono::duration<double>(Clock::now() - p0).count());
        }
        std::sort(pass_seconds.begin(), pass_seconds.end());
        std::printf("%6zu %12.0f %10.1f %10.1f %10.4f\n", ef, double(nq) / pass_seconds[2], percentile(latencies, 0.5),
                    percentile(latencies, 0.99), recall_sum / double(nq));
    }
    return 0;
}