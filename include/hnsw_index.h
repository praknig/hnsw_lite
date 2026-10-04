/**
 * @file hnsw_index.h
 * @brief Approximate nearest-neighbor search with a Hierarchical Navigable
 *        Small World (HNSW) graph.
 */
#pragma once
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "distance.h"
#include "prepared_vector.h"
#include "search_result.h"
#include "storage.h"
#include "visited_list.h"

namespace vecdb {

/**
 * @brief Settings that control HNSW graph quality, memory and speed.
 */
struct HnswParams {
    /// Links per node on levels 1+ (2*M on level 0). Higher: better recall,
    /// more memory, slower inserts. Must be at least 2.
    std::size_t M = 16;
    /// How hard insertion searches for good neighbors. Higher: better graph,
    /// slower inserts. Must be at least 1.
    std::size_t ef_construction = 200;
    /// Seed for random levels. Same seed + same inserts = identical graph.
    std::uint64_t seed = 42;
};

/**
 * @brief Approximate nearest-neighbor index built on a multi-level graph.
 *
 * Structure:
 *  - Every vector is a node on level 0, linked to up to 2*M close nodes.
 *  - Each node also reaches a random top level; about 1 in M nodes reach level 1,
 *    1 in M^2 reach level 2, and so on. Upper levels have few nodes and long links.
 *  - The entry point is a node on the highest level.
 *
 * Search: greedy steps from the entry point down to level 1, then a beam search
 * of width `ef` on level 0. Larger `ef` = more accurate and slower.
 *
 * Insert: pick a random level, store the node, descend greedily to that level,
 * then on each level below find candidates (beam width ef_construction), choose
 * up to M diverse neighbors with the HNSW heuristic, and link both ways.
 *
 * Layers used:
 *  - Layer 1: Storage (VectorStore + GraphStorage + IdMap kept in sync).
 *  - Layer 2: the DistanceFn for `metric`, looked up once in the constructor.
 *
 * Removal: nodes are only flagged. Searches still travel THROUGH removed nodes
 * (so the graph stays connected) but never return them. Inserts also skip them
 * when choosing neighbors, unless every nearby node is removed.
 *
 * Threading: one writer (add/remove) at a time, with no searches running.
 * Several search() calls may run at the same time when no writer is active;
 * each borrows its own VisitedList from a thread-safe pool.
 */
class HnswIndex {
public:
    /// Creates an empty index. Throws std::invalid_argument on invalid params.
    HnswIndex(std::size_t dim, Metric metric, HnswParams params = {});

    /// Inserts a vector under the user's `id`.
    /// Throws std::invalid_argument on a wrong dimension, NaN or infinity, or an
    /// ID already used (IDs cannot be reused, even after remove()); the index
    /// is then unchanged. If memory runs out (std::bad_alloc), the index stays
    /// consistent and `id` is not contained; if the vector was already stored,
    /// its ID stays taken.
    void add(std::uint64_t id, std::span<const float> vector);

    /// Marks `id` as removed. Returns false if it does not exist or was already removed.
    bool remove(std::uint64_t id);

    /// True if `id` is stored and not removed.
    bool contains(std::uint64_t id) const;

    /// Returns up to `k` approximate closest live vectors, closest first.
    /// `ef` is the beam width on level 0; it is raised to at least k and capped
    /// at the number of nodes. A `k` larger than size() is capped at size().
    /// Throws std::invalid_argument if `query` has the wrong dimension or
    /// contains NaN or infinity.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                     std::size_t ef = 64) const;

    /// Number of live (not removed) vectors.
    std::size_t size() const { return live_; }

    std::size_t dim() const { return storage_.vectors().dim(); }
    Metric metric() const { return metric_; }
    const HnswParams& params() const { return params_; }

    /// Highest level in the graph, or -1 when empty.
    int max_level() const { return max_level_; }
    /// Node where every search starts, or kEmpty when empty.
    NodeId entry_point() const { return entry_; }
    /// Read-only access to the underlying storage (used by tests to check the graph).
    const Storage& storage() const { return storage_; }

private:
    /// Draws a random top level: floor(-ln(u) / ln(M)), u uniform in (0, 1].
    int random_level();

    /// Distance from a prepared vector (padded row) to stored node `node`.
    float distance_to(const float* query, NodeId node) const;

    /// Distance between two stored nodes.
    float distance_between(NodeId a, NodeId b) const;

    /// True if two stored nodes hold bit-identical vectors.
    bool same_vector(NodeId a, NodeId b) const;

    /// Greedy walk on one level: moves to any closer neighbor until none is closer.
    Candidate greedy_closest(const float* query, Candidate start, int level) const;

    /// Beam search on one level. Returns up to `ef` closest nodes, closest first.
    /// With `skip_removed`, removed nodes are traversed but not returned.
    std::vector<Candidate> search_level(const float* query, const std::vector<Candidate>& entries,
                                        std::size_t ef, int level, VisitedList& visited,
                                        bool skip_removed) const;

    /// HNSW heuristic: from `sorted` (closest first) keeps a candidate only if it
    /// is closer to the base node than to every already-kept neighbor and is not
    /// an exact copy of one, up to `max`.
    std::vector<NodeId> select_neighbors(const std::vector<Candidate>& sorted,
                                         std::size_t max) const;

    /// Steps 4 to 6 of add(): finds neighbors on every level, links the new node
    /// both ways, and makes it the entry point if it is the highest node.
    void link_new_node(NodeId node, int level);

    /// Sets `node`'s links on `level` and adds the reverse link to each neighbor,
    /// re-selecting a neighbor's list with the heuristic when it is full.
    void connect(NodeId node, const std::vector<NodeId>& neighbors, int level);

    Metric metric_;
    DistanceFn distance_;  // fastest kernel for this CPU, chosen once
    HnswParams params_;
    double level_factor_;  // 1 / ln(M)
    Storage storage_;
    std::size_t stride_;
    PreparedVector scratch_;  // reused by add(); search() uses its own copy
    std::mt19937_64 rng_;
    NodeId entry_ = kEmpty;
    int max_level_ = -1;
    std::size_t live_ = 0;
    mutable VisitedListPool visited_pool_;
};

}  // namespace vecdb