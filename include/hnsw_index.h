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
#include "filter.h"
#include "metadata.h"
#include "prepared_vector.h"
#include "search_result.h"
#include "storage.h"
#include "thread_pool.h"
#include "visited_list.h"

namespace vecdb {

/**
 * @brief Settings that control HNSW graph quality, memory and speed.
 */
struct HnswParams {
    /// Links per node on levels 1+ (2*M on level 0). Higher: better recall,
    /// more memory, slower inserts. Must be at least 2; 8 or more recommended.
    std::size_t M = 16;
    /// How hard insertion searches for good neighbors. Higher: better graph,
    /// slower inserts. Must be at least 1.
    std::size_t ef_construction = 200;
    /// Seed for random levels. Same seed + same operations = identical graph.
    std::uint64_t seed = 42;
    /// Repair the graph around a removed vector: each of its neighbors re-chooses
    /// its links from its own links plus the removed vector's links. Keeps search
    /// quality high under heavy churn; costs some time per remove.
    bool repair_on_remove = true;
};

/**
 * @brief Approximate nearest-neighbor index built on a multi-level graph.
 *
 * Structure:
 *  - Every vector is a node on level 0, linked to up to 2*M close nodes.
 *  - Each node also reaches a random top level; about 1 in M nodes reach level 1,
 *    1 in M^2 reach level 2, and so on. Upper levels have few nodes and long links.
 *  - The entry point is a live node on the highest level.
 *
 * Search: greedy steps from the entry point down to level 1, then a beam search
 * of width `ef` on level 0. Larger `ef` = more accurate and slower.
 *
 * Insert: pick a random level, store the node (reusing the slot of a removed
 * vector if there is one), descend greedily to that level, then on each level
 * below find candidates (beam width ef_construction), choose up to M diverse
 * neighbors with the HNSW heuristic, and link both ways.
 *
 * Real deletion:
 *  - remove() optionally repairs the removed node's neighbors, frees the user ID
 *    (it can be added again) and puts the slot on a free list.
 *  - The next insert reuses a free slot (last in, first out), so memory stays
 *    bounded by the peak number of live vectors.
 *  - Removing the entry point picks a new one: a live node on the highest live
 *    level. Removing every vector leaves the entry point kEmpty and the top
 *    level -1, as in a new index.
 *  - Links from other nodes to a removed or reused slot may remain ("stale"
 *    links). Searches skip any link to a node whose level is below the level
 *    being searched, and inserts and repairs drop stale links they encounter.
 *  - Links are one-directional, so after many removals a search can start in
 *    a region that reaches few live nodes. A search that finds fewer than k
 *    results, or an insert that finds no live neighbor, retries with the entry
 *    point (always live) as an extra starting point.
 *  - compact() rebuilds a dense index from the live vectors.
 *
 * Metadata and filters:
 *  - Each vector can carry metadata (MetadataStore); it follows the vector's
 *    slot through reuse, removal and compact().
 *  - A filtered search asks the query planner (plan_search) whether to scan the
 *    matching vectors exactly or search the graph. A graph search treats
 *    vectors failing the filter like removed ones: traveled through, never
 *    returned. With no filter, eligibility is exactly "not removed".
 *  - search_batch() runs queries on a thread pool; search_range() grows a
 *    region from the nearest vectors outward through graph links.
 *
 * Layers used:
 *  - Layer 1: Storage (VectorStore + GraphStorage + IdMap kept in sync).
 *  - Layer 2: the DistanceFn for `metric`, looked up once in the constructor.
 *
 * Threading: one writer (add/remove/compact) at a time, with no searches
 * running. Several search() calls may run at the same time when no writer is
 * active; each borrows its own VisitedList from a thread-safe pool.
 */
class HnswIndex {
public:
    /// Creates an empty index. `schema` defines the metadata fields (dynamic by
    /// default). Throws std::invalid_argument on invalid params.
    HnswIndex(std::size_t dim, Metric metric, HnswParams params = {}, Schema schema = Schema::dynamic());

    /// Inserts a vector, with optional metadata, under the user's `id`.
    /// Throws std::invalid_argument on a wrong dimension, NaN or infinity, an
    /// ID that is currently stored, or invalid metadata; the index is then unchanged. If memory runs
    /// out (std::bad_alloc), the index stays consistent and `id` is not stored.
    void add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata = {});

    /// Deletes `id` for real: repairs the graph around it (if enabled), frees
    /// the ID for reuse and puts its slot on the free list. Returns false if the
    /// ID is not stored. If memory runs out during repair, the vector stays
    /// stored and the graph stays valid.
    bool remove(std::uint64_t id);

    /// True if `id` is stored.
    bool contains(std::uint64_t id) const;

    /// Returns up to `k` approximate closest vectors, closest first.
    /// `ef` is the beam width on level 0; it is raised to at least k and capped
    /// at the number of nodes. A `k` larger than size() is capped at size().
    /// Throws std::invalid_argument if `query` has the wrong dimension or
    /// contains NaN or infinity.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                     std::size_t ef = 64) const;

    /// Same, with a filter, predicate, strategy and statistics (see SearchOptions).
    /// With a filter or predicate, the planner chooses exact or graph search.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t ef,
                                     const SearchOptions& options) const;

    /// Searches many queries at once (`dim` floats each, back to back), on
    /// `threads` threads (0 = one per hardware thread). Results are identical to
    /// separate search() calls. Every query is validated before any work.
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k,
                                                        std::size_t ef = 64, std::size_t threads = 0,
                                                        const SearchOptions& options = {}) const;

    /// Approximate: every eligible vector with distance <= `radius` (for L2, a
    /// squared distance: see l2_radius()), closest first, at most `max_results`.
    /// Grows a region from the nearest vectors through graph links; see
    /// SearchOptions::range_expand_outside. Strategy::ForceExact scans instead.
    /// Throws std::invalid_argument for a NaN radius.
    std::vector<SearchResult> search_range(std::span<const float> query, float radius,
                                           std::size_t max_results = kNoLimit,
                                           const SearchOptions& options = {}) const;

    /// Replaces the given metadata fields of `id` (unset() clears one). Returns
    /// false if `id` is not stored. Throws std::invalid_argument on invalid
    /// metadata. All-or-nothing; the vector and graph are untouched.
    bool set_metadata(std::uint64_t id, const Metadata& metadata);

    /// The metadata of `id`, or nothing if `id` is not stored.
    std::optional<Metadata> get_metadata(std::uint64_t id) const;

    /// Keeps exact value counts for a keyword, boolean or tag-set field, so the
    /// planner can count matches instead of sampling. Throws std::invalid_argument otherwise.
    void create_payload_index(std::string_view field) { metadata_.index_field(field); }

    /// The index's planner settings (per-query settings in SearchOptions override them).
    void set_planner_params(const PlannerParams& p) {
        validate(p);
        planner_ = p;
    }
    const PlannerParams& planner_params() const { return planner_; }
    const MetadataStore& metadata() const { return metadata_; }
    /// Visited lists kept for reuse by searches (released by compact()).
    std::size_t pooled_visited_lists() const { return visited_pool_.idle_count(); }

    /// Rebuilds the index from the live vectors only, with dense internal
    /// numbers and no free slots. User IDs are kept. The new index is built
    /// completely before replacing the old one, so if memory runs out the old
    /// index is left intact.
    CompactStats compact();

    /// Number of live vectors.
    std::size_t size() const { return live_; }
    /// Slots of removed vectors still occupying memory (waiting to be reused).
    std::size_t deleted_count() const { return storage_.size() - live_; }
    /// All slots, live or removed.
    std::size_t capacity() const { return storage_.size(); }

    std::size_t dim() const { return storage_.vectors().dim(); }
    Metric metric() const { return metric_; }
    const HnswParams& params() const { return params_; }

    /// Highest level of a live node, or -1 when the index has no live vectors.
    int max_level() const { return max_level_; }
    /// Node where every search starts, or kEmpty when there are no live vectors.
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

    /// Greedy walk on one level: moves to any closer neighbor until none is
    /// closer. Never steps onto `exclude` or onto a node below `level`.
    Candidate greedy_closest(const float* query, Candidate start, int level, NodeId exclude) const;

    /// Beam search on one level. Returns up to `ef` closest eligible nodes,
    /// closest first. Ineligible nodes (removed, or failing the filter) are
    /// traversed but never returned. `exclude` (a node being inserted) is never
    /// visited. Counts work in `stats` if given.
    std::vector<Candidate> search_level(const float* query, const std::vector<Candidate>& entries,
                                        std::size_t ef, int level, VisitedList& visited,
                                        NodeId exclude, const Eligibility& eligible,
                                        SearchStats* stats = nullptr) const;

    /// Eligibility with no filter: "not removed".
    Eligibility live_only() const { return Eligibility{&storage_.ids()}; }

    /// Compiles the options' filter (if any) into `holder` and returns the eligibility test.
    Eligibility eligibility(const SearchOptions& options, std::optional<CompiledFilter>& holder) const;

    /// The search itself for one prepared query: plans, then runs exact or graph search.
    std::vector<SearchResult> search_impl(const float* query, std::size_t k, std::size_t ef,
                                          const Eligibility& eligible, Strategy strategy,
                                          const PlannerParams& params, SearchStats& stats) const;

    /// Exact top-k over every eligible node.
    std::vector<Candidate> exact_scan(const float* query, std::size_t k, const Eligibility& eligible,
                                      SearchStats& stats) const;

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

    /// Before `node` is removed: each live neighbor re-chooses its links from
    /// its own links plus `node`'s links, minus `node`, on every level.
    void repair_neighbors(NodeId node);

    /// Makes the live node with the highest level the entry point, or sets the
    /// entry point to kEmpty and the top level to -1 if no node is live.
    void choose_new_entry();

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
    std::vector<NodeId> free_slots_;  // slots of removed vectors, reused last in first out
    MetadataStore metadata_;
    PlannerParams planner_;
    mutable VisitedListPool visited_pool_;
    mutable SharedPool pool_;
};

}  // namespace vecdb