/**
 * @file health.h
 * @brief Health reports: measurements of an index's state after many changes,
 *        so that upkeep (like a database's vacuum) can be decided on evidence.
 *
 * A report is read-only and cheap except for the recall estimate, which runs
 * `recall_sample` searches and as many exact scans (set it to 0 to skip it).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vecdb {

struct HealthOptions {
    /// Stored vectors searched for to estimate recall (0: no estimate). Each costs
    /// one search and one exact scan over the live vectors.
    std::size_t recall_sample = 100;
    std::size_t k = 10;          ///< recall@k
    std::size_t ef = 64;         ///< search beam for the estimate (as search's default)
    std::uint64_t seed = 1;      ///< which vectors are sampled
    /// Stored vectors on which compression and IVF figures are measured (0: none).
    /// Needs the stored vectors (not codes only).
    std::size_t codec_sample = 500;
    /// Stored vectors (apart from those above) on which the codec and the IVF
    /// centroids are retrained, in memory, to show what retraining would gain
    /// (0: no comparison). Costs a training (k-means for PQ and IVF).
    std::size_t retrain_sample = 0;
};

/// One level of an HNSW graph: live nodes present on it, and how many the
/// random level assignment would place there (live * M^-level).
struct LevelHealth {
    std::size_t nodes = 0;
    double expected = 0;
};

struct HealthReport {
    // Storage.
    std::size_t live = 0;             ///< vectors stored
    std::size_t slots = 0;            ///< slots in use or free
    std::size_t free_slots = 0;       ///< slots of removed vectors, waiting for reuse
    // The graph (level 0 unless stated; searches end on level 0).
    std::vector<LevelHealth> levels;  ///< per level, from 0 to the top
    double mean_degree = 0;           ///< mean neighbors of a live node
    std::size_t min_degree = 0;       ///< the fewest neighbors of a live node
    std::size_t underfull = 0;        ///< live nodes with fewer than M neighbors (half the usual 2M)
    std::size_t no_incoming = 0;      ///< live nodes no live node links to (fragile: reached only through removed nodes, if at all)
    std::size_t unreachable = 0;      ///< live nodes a search from the entry point cannot reach (it may pass removed nodes)
    std::size_t links_to_removed = 0; ///< links from live nodes to removed ones (searches still follow them: wasted work)
    std::size_t stale_links = 0;      ///< links, on any level, to a node that is not on that level
    // Compression (with a trained codec and the stored vectors kept).
    /// Reconstruction error on a sample: squared distance between each vector and
    /// its decoded code, relative to the vectors' squared length (0 = exact).
    std::optional<double> codec_error;
    /// The same error with the codec retrained on a current sample (retrain_sample):
    /// much lower than codec_error means the data has drifted since training.
    std::optional<double> codec_error_retrained;
    /// SQ8: the share of values outside the trained range (clipped when encoded).
    std::optional<double> clipped;
    // IVF (FlatIndex with lists).
    std::size_t lists = 0;            ///< configured lists (0: no IVF)
    bool lists_trained = false;
    std::size_t largest_list = 0;     ///< vectors in the largest list
    std::size_t empty_lists = 0;
    /// The largest list relative to the mean (1 = perfectly balanced); probing a
    /// few lists scans a share of the vectors that grows with this.
    double list_imbalance = 0;
    /// Mean squared distance of sampled vectors to their list's centroid.
    std::optional<double> centroid_distance;
    /// The same with centroids retrained on a current sample (retrain_sample):
    /// much lower means the lists no longer fit the data.
    std::optional<double> centroid_distance_retrained;
    // Quality.
    /// Estimated recall@k of searches for stored vectors against exact answers;
    /// none if not requested, or if only codes are stored (no exact answers).
    std::optional<double> recall;
    // Changes since the last upkeep (since construction or loading until upkeep exists).
    std::size_t removals = 0;         ///< vectors removed
    std::size_t updates = 0;          ///< vectors replaced (update, upsert of an existing ID)
    /// Removals and updates relative to the live count: the share of the graph disturbed.
    double churn() const { return double(removals + updates) / double(live > 0 ? live : 1); }
};

}  // namespace vecdb