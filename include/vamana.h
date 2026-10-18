/**
 * @file vamana.h
 * @brief The Vamana graph (from DiskANN): a single-layer navigable graph built
 *        for searches that visit few nodes, which is what a disk-based index needs.
 *
 * Building (Subramanya et al., "DiskANN", NeurIPS 2019): start from a random
 * graph of degree R; then, for every node in a random order, search for it
 * from the medoid with a beam of L, and replace its neighbors by the "robust
 * pruning" of everything visited: keep the closest candidate, drop every
 * candidate that is alpha times closer to it than to the node, and repeat,
 * up to R neighbors; add the reverse edges, pruning any node that overflows.
 * Two passes: alpha = 1, then the given alpha (1.2 by default), which keeps
 * long-range edges and so shortens searches.
 *
 * Two deviations, both for degenerate data: with at most R + 1 vectors the
 * complete graph is kept (pruning could only remove useful edges), and a
 * candidate identical to the node (distance 0) is never pruned (otherwise
 * identical vectors would keep one neighbor each).
 *
 * Building ends with a connectivity repair: Vamana does not guarantee every
 * node an incoming edge (rarely, one is never chosen as a neighbor), and on
 * disk an unreachable vector could never be returned. Each unreachable node
 * gets an edge from the nearest reachable node found by searching for it
 * (replacing that node's farthest neighbor if its list is full), repeated
 * until every node is reachable (at most 16 rounds). Degrees stay at most R.
 *
 * This class builds and searches in memory; the disk-based index (Q6) writes
 * its graph to disk. Deterministic for a given seed. L2 and cosine only.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "common.h"
#include "distance.h"
#include "search_result.h"
#include "vector_store.h"

namespace vecdb {

struct VamanaParams {
    std::size_t R = 64;        ///< the largest number of neighbors per node (at least 1)
    std::size_t L = 100;       ///< the beam width while building (at least R)
    float alpha = 1.2f;        ///< pruning strength of the second pass (at least 1)
    std::uint64_t seed = 1;    ///< same seed, same graph
};

/// Throws std::invalid_argument for settings that cannot work.
void validate(const VamanaParams& p);

class VamanaGraph {
public:
    /// Builds the graph over `vectors` (n * dim values; normalized here for
    /// cosine). Throws std::invalid_argument for no vectors, a size that is
    /// not a multiple of dim, non-finite values, invalid settings, or a metric
    /// other than L2 and cosine.
    static VamanaGraph build(std::span<const float> vectors, std::size_t dim, Metric metric, const VamanaParams& params = {});

    std::size_t size() const { return vectors_.size(); }
    std::size_t dim() const { return vectors_.dim(); }
    Metric metric() const { return metric_; }
    const VamanaParams& params() const { return params_; }
    /// The entry point of every search: the vector nearest to the mean.
    NodeId medoid() const { return medoid_; }
    std::span<const NodeId> neighbors(NodeId node) const { return adjacency_.at(node); }
    /// The stored (prepared) vector of a node.
    std::span<const float> vector(NodeId node) const { return vectors_.get(node); }

    /// Beam search from the medoid: the k nearest nodes (ids are node numbers),
    /// with a beam of max(L, k). `expanded`, if given, receives how many nodes
    /// were expanded (on disk, one read each).
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t L,
                                     std::size_t* expanded = nullptr) const;

private:
    VamanaGraph(std::size_t dim, Metric metric, const VamanaParams& params);
    float distance(const float* a, NodeId b) const;
    /// Beam search from `start` for `query` (padded); returns the expanded nodes
    /// with their distances, and leaves the best `beam` candidates in `best`.
    std::vector<std::pair<float, NodeId>> greedy(const float* query, std::size_t beam,
                                                 std::vector<std::pair<float, NodeId>>& best) const;
    std::vector<NodeId> robust_prune(NodeId p, std::vector<std::pair<float, NodeId>> candidates, float alpha) const;
    std::vector<bool> reachable() const;  // from the medoid
    void repair_connectivity();          // give every node a path from the medoid

    VectorStore vectors_;
    Metric metric_;
    DistanceFn distance_;
    VamanaParams params_;
    NodeId medoid_ = 0;
    std::vector<std::vector<NodeId>> adjacency_;
    mutable std::vector<std::uint32_t> mark_;  // visited generation per node (searches are not concurrent while building)
    mutable std::uint32_t generation_ = 0;
};

}  // namespace vecdb