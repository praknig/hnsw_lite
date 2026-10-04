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
 *  - search() computes the distance from the query to every vector and keeps
 *    the k closest in a TopK heap.
 *  - remove() deletes for real with swap-with-last: the last vector is moved
 *    into the removed vector's slot. No tombstones, constant time, and the
 *    removed ID can be added again immediately.
 *
 * Layers used:
 *  - Layer 1: VectorStore and IdMap directly. Not Storage, because Storage would
 *    also allocate HNSW neighbor slots that a flat index never uses.
 *  - Layer 2: the DistanceFn for `metric`, looked up once in the constructor.
 *
 * Guarantees:
 *  - Results are exact, sorted closest first. Equal distances are ordered by
 *    internal position, which is insertion order until a removal moves a vector.
 *  - Invalid input (wrong dimension, NaN, infinity, duplicate ID) throws and
 *    changes nothing; so does running out of memory.
 *
 * Threading: one writer (add/remove) at a time, with no searches running.
 * Several search() calls may run at the same time when no writer is active.
 */
class FlatIndex {
public:
    /// Creates an empty index for vectors of `dim` floats.
    FlatIndex(std::size_t dim, Metric metric);

    /// Stores a vector under the user's `id`. Throws std::invalid_argument on a
    /// wrong dimension, NaN or infinity, or an ID that is currently stored.
    void add(std::uint64_t id, std::span<const float> vector);

    /// Deletes `id` for real (swap-with-last). Returns false if it is not stored.
    /// The ID can be added again afterwards. Never allocates.
    bool remove(std::uint64_t id);

    /// True if `id` is stored.
    bool contains(std::uint64_t id) const;

    /// Returns up to `k` closest vectors, closest first. A `k` larger than
    /// size() returns every vector. Throws std::invalid_argument if `query` has
    /// the wrong dimension or contains NaN or infinity.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

    /// Does nothing (swap-with-last never leaves holes) and reports that.
    CompactStats compact() { return {live_, 0}; }

    /// Number of stored vectors.
    std::size_t size() const { return live_; }
    /// Removed vectors still occupying memory: always 0 for a flat index.
    std::size_t deleted_count() const { return 0; }
    /// Slots in use, removed or not: always equal to size() for a flat index.
    std::size_t capacity() const { return vectors_.size(); }

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