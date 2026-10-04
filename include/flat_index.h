/**
 * @file flat_index.h
 * @brief Exact brute-force search: compares the query with every stored vector.
 */
#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "distance.h"
#include "id_map.h"
#include "prepared_vector.h"
#include "search_result.h"
#include "vector_store.h"

namespace vecdb {

/**
 * @brief Exact nearest-neighbor index ("flat" = no structure, just one list).
 *
 * How it works:
 *  - add() normalizes the vector if needed (PreparedVector), registers the
 *    user ID (IdMap) and stores the values (VectorStore).
 *  - search() computes the distance from the query to every live vector and
 *    keeps the k closest in a TopK heap.
 *
 * Layers used:
 *  - Layer 1: VectorStore and IdMap directly. Not Storage, because Storage would
 *    also allocate HNSW neighbor slots that a flat index never uses.
 *  - Layer 2: the DistanceFn for `metric`, looked up once in the constructor.
 *
 * Guarantees:
 *  - Results are exact, sorted closest first; equal distances are ordered by
 *    insertion order.
 *  - Removed vectors never appear in results.
 *  - Invalid input (wrong dimension, duplicate ID) throws and changes nothing.
 *
 * Threading: one writer (add/remove) at a time, with no searches running.
 * Several search() calls may run at the same time when no writer is active.
 *
 * Cost: one distance per stored vector per search. Best for up to roughly
 * 100,000 vectors, and used as the ground truth when measuring HNSW recall.
 */
class FlatIndex {
public:
    /// Creates an empty index for vectors of `dim` floats.
    FlatIndex(std::size_t dim, Metric metric);

    /// Stores a vector under the user's `id`.
    /// Throws std::invalid_argument on a wrong dimension or an ID already used
    /// (IDs cannot be reused, even after remove()).
    void add(std::uint64_t id, std::span<const float> vector);

    /// Marks `id` as removed. Returns false if it does not exist or was already removed.
    bool remove(std::uint64_t id);

    /// True if `id` is stored and not removed.
    bool contains(std::uint64_t id) const;

    /// Returns up to `k` closest live vectors, closest first.
    /// Throws std::invalid_argument if `query` has the wrong dimension.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

    /// Number of live (not removed) vectors.
    std::size_t size() const { return live_; }

    std::size_t dim() const { return vectors_.dim(); }
    Metric metric() const { return metric_; }

private:
    Metric metric_;
    DistanceFn distance_;     // fastest kernel for this CPU, chosen once
    VectorStore vectors_;
    IdMap ids_;
    PreparedVector scratch_;  // reused by add(); search() uses its own copy
    std::size_t live_ = 0;
};

}  // namespace vecdb