/**
 * @file kmeans.cpp
 * @brief k-means++ seeding and Lloyd's refinement; see kmeans.h.
 *
 * Built without floating-point contraction (see CMakeLists.txt), so results are
 * bit-identical on every CPU.
 */
#include "kmeans.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "thread_pool.h"

namespace vecdb {

namespace {

/// SplitMix64: portable, so the same seed gives the same choices everywhere.
struct Random {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }  // [0, 1)
};

/// Squared L2 distance, accumulated in double in a fixed order.
double squared_distance(const float* a, const float* b, std::size_t dim) {
    double sum = 0;
    for (std::size_t d = 0; d < dim; ++d) {
        const double diff = double(a[d]) - double(b[d]);
        sum += diff * diff;
    }
    return sum;
}

constexpr std::size_t kChunk = 1024;  // points per assignment task: fixed, so threads never change results

}  // namespace

std::uint32_t nearest_centroid(std::span<const float> centroids, std::size_t dim, const float* point, float* distance) {
    const std::size_t k = centroids.size() / dim;
    std::uint32_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < k; ++c) {
        const double d = squared_distance(point, centroids.data() + c * dim, dim);
        if (d < best_d) {  // strict: ties keep the lower index
            best_d = d;
            best = static_cast<std::uint32_t>(c);
        }
    }
    if (distance) *distance = static_cast<float>(best_d);
    return best;
}

KMeansResult kmeans(std::span<const float> data, std::size_t dim, const KMeansParams& params) {
    if (dim == 0) throw std::invalid_argument("kmeans: dimension must be at least 1");
    if (data.size() % dim != 0) throw std::invalid_argument("kmeans: data size is not a multiple of the dimension");
    const std::size_t n = data.size() / dim, k = params.k;
    if (n == 0) throw std::invalid_argument("kmeans: no points");
    if (k == 0 || k > n) throw std::invalid_argument("kmeans: k must be between 1 and the number of points");
    if (params.iterations == 0) throw std::invalid_argument("kmeans: iterations must be at least 1");
    for (float x : data)
        if (!std::isfinite(x)) throw std::invalid_argument("kmeans: data contains NaN or infinity");
    auto point = [&](std::size_t i) { return data.data() + i * dim; };

    KMeansResult r;
    r.centroids.resize(k * dim);
    r.assignment.assign(n, 0);
    auto centroid = [&](std::size_t c) { return r.centroids.data() + c * dim; };

    // 1. k-means++ seeding: each new centroid is a point drawn with probability
    //    proportional to its squared distance from the nearest centroid so far.
    Random rng{params.seed};
    std::vector<double> nearest(n);
    std::size_t first = static_cast<std::size_t>(rng.next() % n);
    std::copy(point(first), point(first) + dim, centroid(0));
    for (std::size_t i = 0; i < n; ++i) nearest[i] = squared_distance(point(i), centroid(0), dim);
    for (std::size_t c = 1; c < k; ++c) {
        double total = 0;
        for (double d : nearest) total += d;
        std::size_t pick = c % n;  // every point already coincides with a centroid: take any
        if (total > 0) {
            const double target = rng.uniform() * total;
            double running = 0;
            pick = n - 1;
            for (std::size_t i = 0; i < n; ++i) {
                running += nearest[i];
                if (running > target && nearest[i] > 0) {
                    pick = i;
                    break;
                }
            }
        }
        std::copy(point(pick), point(pick) + dim, centroid(c));
        for (std::size_t i = 0; i < n; ++i) nearest[i] = std::min(nearest[i], squared_distance(point(i), centroid(c), dim));
    }

    // 2. Lloyd's refinement: assign every point to its nearest centroid, then move
    //    each centroid to the mean of its points, until nothing changes.
    std::vector<float> dist(n);
    const std::size_t threads = std::min(effective_threads(params.threads), (n + kChunk - 1) / kChunk);
    std::unique_ptr<ThreadPool> pool = threads > 1 ? std::make_unique<ThreadPool>(threads) : nullptr;
    auto assign = [&](std::vector<std::uint32_t>& into) {
        auto chunk = [&](std::size_t t) {
            const std::size_t end = std::min(n, (t + 1) * kChunk);
            for (std::size_t i = t * kChunk; i < end; ++i) into[i] = nearest_centroid(r.centroids, dim, point(i), &dist[i]);
        };
        const std::size_t chunks = (n + kChunk - 1) / kChunk;
        if (pool) pool->parallel_for(chunks, chunk);
        else for (std::size_t t = 0; t < chunks; ++t) chunk(t);
    };

    std::vector<std::uint32_t> next(n);
    std::vector<double> sums(k * dim);
    std::vector<std::size_t> counts(k);
    assign(r.assignment);
    for (std::size_t it = 1; it <= params.iterations; ++it) {
        r.iterations_run = it;
        // Means of the current clusters, summed in point order.
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c = r.assignment[i];
            ++counts[c];
            for (std::size_t d = 0; d < dim; ++d) sums[c * dim + d] += point(i)[d];
        }
        // An empty cluster takes the point farthest from its own centroid (from a
        // cluster that keeps at least one point; ties go to the lower index).
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] != 0) continue;
            std::size_t far = n;
            for (std::size_t i = 0; i < n; ++i)
                if (counts[r.assignment[i]] > 1 && (far == n || dist[i] > dist[far])) far = i;
            if (far == n) continue;  // GCOVR_EXCL_LINE: cannot happen while k <= n (some cluster has 2+ points); kept for safety
            const std::size_t from = r.assignment[far];
            --counts[from];
            ++counts[c];
            for (std::size_t d = 0; d < dim; ++d) {
                sums[from * dim + d] -= point(far)[d];
                sums[c * dim + d] += point(far)[d];
            }
            r.assignment[far] = static_cast<std::uint32_t>(c);
            dist[far] = 0;
        }
        for (std::size_t c = 0; c < k; ++c)
            for (std::size_t d = 0; d < dim; ++d) centroid(c)[d] = static_cast<float>(sums[c * dim + d] / double(counts[c]));
        assign(next);
        const bool stable = next == r.assignment;
        r.assignment.swap(next);
        if (stable) break;
    }
    r.inertia = 0;
    for (float d : dist) r.inertia += d;
    return r;
}

}  // namespace vecdb