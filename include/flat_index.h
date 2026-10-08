/**
 * @file flat_index.h
 * @brief Exact brute-force search: compares the query with every stored vector.
 */
#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "distance.h"
#include "filter.h"
#include "id_map.h"
#include "metadata.h"
#include "prepared_vector.h"
#include "search_result.h"
#include "thread_pool.h"
#include "vector_store.h"

namespace vecdb {

/**
 * @brief Exact nearest-neighbor index ("flat" = no structure, just one list).
 *
 * How it works:
 *  - add() normalizes the vector if needed (PreparedVector), registers the
 *    user ID (IdMap), stores the values (VectorStore) and the metadata
 *    (MetadataStore), all-or-nothing.
 *  - search() computes the distance from the query to every eligible vector
 *    (passing the filter and predicate, if any) and keeps the k closest.
 *  - remove() deletes for real with swap-with-last: the last vector and its
 *    metadata move into the removed vector's slot. No tombstones, and the
 *    removed ID can be added again immediately.
 *  - search_batch() compares blocks of stored vectors with blocks of queries
 *    (tiling), so each vector loaded into the cache serves many queries.
 *  - search_range() returns every eligible vector within a radius.
 *
 * Results are exact, sorted closest first; equal distances are ordered by
 * internal position (insertion order until a removal moves a vector).
 *
 * Threading: one writer at a time, with no searches running. Searches (also
 * batch and range) may run at the same time when no writer is active.
 */
class FlatIndex {
public:
    /// Creates an empty index for vectors of `dim` floats. `schema` defines the
    /// metadata fields (dynamic by default: fields are created on first use).
    FlatIndex(std::size_t dim, Metric metric, Schema schema = Schema::dynamic());

    /// Stores a vector, with optional metadata, under the user's `id`. Throws
    /// std::invalid_argument on a wrong dimension, NaN or infinity, an ID that is
    /// currently stored, or invalid metadata. All-or-nothing.
    void add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata = {});

    /// Deletes `id` for real (swap-with-last). Returns false if it is not stored.
    bool remove(std::uint64_t id);

    /// Replaces the vector stored under `id` in place, keeping its metadata.
    /// Returns false if `id` is not stored. Throws std::invalid_argument on a
    /// wrong dimension, NaN or infinity (checked first, even for an unknown ID).
    /// All-or-nothing.
    bool update(std::uint64_t id, std::span<const float> vector);

    /// Replaces the vector and its whole metadata (fields not given are cleared;
    /// to change some fields only, use set_metadata()). All-or-nothing.
    bool update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);

    /// Adds `id` if it is not stored, otherwise replaces its vector (keeping its
    /// metadata). Returns true if it was added, false if it was replaced.
    bool upsert(std::uint64_t id, std::span<const float> vector);

    /// Adds `id` with this metadata, or replaces its vector and whole metadata.
    /// Returns true if it was added, false if it was replaced.
    bool upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);

    /// True if `id` is stored.
    bool contains(std::uint64_t id) const;

    /// Replaces the given metadata fields of `id` (unset() clears one); other
    /// fields keep their values. Returns false if `id` is not stored. Throws
    /// std::invalid_argument on invalid metadata. All-or-nothing.
    bool set_metadata(std::uint64_t id, const Metadata& metadata);

    /// The metadata of `id`, or nothing if `id` is not stored.
    std::optional<Metadata> get_metadata(std::uint64_t id) const;

    /// Keeps exact value counts for a keyword, boolean or tag-set field (used by
    /// filtered searches' statistics). Throws std::invalid_argument otherwise.
    void create_payload_index(std::string_view field) { metadata_.index_field(field); }

    /// Returns up to `k` closest vectors, closest first. A `k` larger than
    /// size() returns every vector. Throws std::invalid_argument if `query` has
    /// the wrong dimension or contains NaN or infinity.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

    /// Same, with a filter, predicate and statistics (see SearchOptions). A flat
    /// search is always exact, so the strategy setting has no effect.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, const SearchOptions& options) const;

    /// Searches many queries at once (`dim` floats each, back to back), on
    /// `threads` threads (0 = one per hardware thread). Results are identical to
    /// separate search() calls. Every query is validated before any work.
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k,
                                                        std::size_t threads = 0,
                                                        const SearchOptions& options = {}) const;

    /// Every eligible vector with distance <= `radius` (for L2, a squared
    /// distance: see l2_radius()), closest first, at most `max_results`.
    /// Throws std::invalid_argument for a NaN radius.
    std::vector<SearchResult> search_range(std::span<const float> query, float radius,
                                           std::size_t max_results = kNoLimit,
                                           const SearchOptions& options = {}) const;

    /// Swap-with-last never leaves holes in the vectors, so this only reclaims
    /// metadata strings no vector uses any more. Reports 0 vectors reclaimed.
    CompactStats compact() {
        metadata_.compact_dictionary();
        return {live_, 0};
    }

    std::size_t size() const { return live_; }
    std::size_t deleted_count() const { return 0; }
    std::size_t capacity() const { return vectors_.size(); }
    std::size_t dim() const { return vectors_.dim(); }
    Metric metric() const { return metric_; }
    const MetadataStore& metadata() const { return metadata_; }

private:
    /// update() and upsert(): `metadata` null keeps the old metadata.
    bool replace(std::uint64_t id, std::span<const float> vector, const Metadata* metadata);

    /// Compiles the options' filter (if any) into `holder` and returns the eligibility test.
    Eligibility eligibility(const SearchOptions& options, std::optional<CompiledFilter>& holder) const;

    /// Exact top-k over eligible vectors for one prepared query.
    std::vector<SearchResult> scan(const float* query, std::size_t k, const Eligibility& eligible,
                                   SearchStats* stats) const;

    Metric metric_;
    DistanceFn distance_;     // fastest kernel for this CPU, chosen once
    VectorStore vectors_;
    IdMap ids_;
    MetadataStore metadata_;
    PreparedVector scratch_;  // reused by add(); searches use their own copy
    std::size_t live_ = 0;
    mutable SharedPool pool_;
};

}  // namespace vecdb