/**
 * @file kmeans.h
 * @brief k-means clustering, the training step shared by product quantization
 *        and the IVF index.
 *
 * Deterministic: the same data, k, iterations and seed give bit-identical
 * centroids on every CPU, compiler and thread count. This matters because
 * centroids are saved in index files, and the golden-file tests require those
 * files to be identical on every platform. To achieve it:
 *  - the random choices use SplitMix64 (portable, unlike std:: distributions);
 *  - each point's assignment is independent of the others, so threads only
 *    change who computes it, never the result;
 *  - sums are accumulated in a fixed order, in double precision;
 *  - kmeans.cpp is compiled without floating-point contraction, so a*b + c is
 *    never fused into one instruction (GCC does that by default on ARM, which
 *    would change rounding there).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecdb {

struct KMeansParams {
    std::size_t k = 256;           ///< number of centroids, 1..number of points
    std::size_t iterations = 25;   ///< upper bound on refinement rounds (at least 1)
    std::uint64_t seed = 1;        ///< same seed, same result
    std::size_t threads = 1;       ///< for assigning points; 0 means all cores. Never changes the result
};

struct KMeansResult {
    std::vector<float> centroids;          ///< k * dim values, centroid by centroid
    std::vector<std::uint32_t> assignment; ///< the nearest centroid of each point
    double inertia = 0;                    ///< sum of squared distances to the assigned centroids
    std::size_t iterations_run = 0;        ///< rounds actually run (stops early once stable)
};

/// Clusters `data` (n * dim values, point by point) into params.k groups with
/// k-means++ seeding and Lloyd's refinement, using squared L2 distance.
/// A cluster that becomes empty takes the point farthest from its own centroid.
/// Throws std::invalid_argument for dim 0, a size that is not a multiple of dim,
/// no points, k of 0 or above the number of points, 0 iterations, or a value
/// that is NaN or infinite.
KMeansResult kmeans(std::span<const float> data, std::size_t dim, const KMeansParams& params);

/// The index of the centroid nearest to `point` (squared L2; ties go to the
/// lower index). `centroids` holds k * dim values.
std::uint32_t nearest_centroid(std::span<const float> centroids, std::size_t dim, const float* point,
                               float* distance = nullptr);

}  // namespace vecdb