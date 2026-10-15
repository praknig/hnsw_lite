/**
 * @file flat_index.h
 * @brief Exact brute-force search: compares the query with every stored vector.
 */
#pragma once
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
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

    /// Adds many vectors at once. `vectors` holds ids.size() vectors back to
    /// back; `metadata`, if given, has one entry per vector. The index's lock is
    /// held throughout, so searches wait and then see all of the batch or none of it.
    ///
    /// **All or nothing.** Every vector, ID (against the index and within the
    /// batch) and metadata entry is checked before anything changes; any problem
    /// throws std::invalid_argument and leaves the index unchanged.
    /// If memory runs out partway, every vector added so far is taken out
    /// again, and the index is exactly as before. Inserting into a Flat index
    /// is a copy, so it runs on the calling thread; `threads` is accepted for
    /// symmetry with HnswIndex.
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads = 0);
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                   std::span<const Metadata> metadata, std::size_t threads = 0);

    /// The vector stored for `id`, exactly as it was given to add() or update()
    /// (for cosine indexes, before normalization: the original is kept for this),
    /// or nothing if `id` is not stored.
    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;

    /// The vector and metadata of `id` together, or nothing if it is not stored.
    std::optional<Record> get(std::uint64_t id) const;

    /// Lists stored records a page at a time, in ascending ID order, optionally
    /// only those matching a filter (same rules as count_where). Each page
    /// continues after the last ID of the previous one (ScanOptions::after =
    /// Page::next), so changes between pages never repeat or skip a record that
    /// stays stored. Each page costs one pass over the index. Throws
    /// std::invalid_argument for a limit of 0.
    Page scan(const ScanOptions& options = {}) const;

    /// Removes every vector whose metadata matches `filter` (like SQL's DELETE
    /// WHERE) and returns how many it removed. The filter is evaluated once,
    /// before anything changes. A filter naming an unknown field throws
    /// std::invalid_argument under a strict schema (nothing changes) and matches
    /// nothing under a dynamic one. Searches see the index before or after, never
    /// in between.
    ///
    /// **Always complete:** once the filter is accepted, every match is removed
    /// (removal never allocates).
    std::size_t remove_where(const Filter& filter);

    /// Merges `changes` into the metadata of every vector matching `filter`, as
    /// set_metadata() does for one (Metadata::unset() removes a field), and
    /// returns how many it changed. The filter is evaluated once, before
    /// anything changes, so a change to the very field it tests applies to each
    /// matching vector exactly once.
    ///
    /// **All or nothing:** invalid changes throw std::invalid_argument, even if
    /// nothing matches; if memory runs out, no vector has changed. To achieve
    /// this the change is made on a copy of the metadata, swapped in when
    /// complete, so it needs extra memory about the size of the index's metadata
    /// while it runs.
    std::size_t set_metadata_where(const Filter& filter, const Metadata& changes);

    /// How many vectors match `filter`: a preview of remove_where or
    /// set_metadata_where. Same filter rules as those.
    std::size_t count_where(const Filter& filter) const;

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
    void create_payload_index(std::string_view field) {
        const auto guard = lock_.write("create_payload_index");
        metadata_.index_field(field);
    }

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
        const auto guard = lock_.write("compact");
        metadata_.compact_dictionary();
        return {live_, 0};
    }

    /// Saves the whole index (vectors, graph, metadata, settings) to `path`. It is
    /// written to a temporary file next to it ("<path>.tmp-" and a unique
    /// suffix) and then renamed into place, so a failure never damages an
    /// existing file at `path`, and saves to one path at the same time each
    /// produce a complete file (the last one wins). Searches may run during a save;
    /// changes attempted during one throw std::logic_error. Throws
    /// std::runtime_error if the file cannot be written.
    void save(const std::string& path) const;

    /// Saves to a stream (for example a std::ostringstream).
    void save(std::ostream& out) const;

    /// Loads an index saved by save(). Throws FormatError if the file is damaged,
    /// truncated, from an unsupported version or of the other index kind (a
    /// failed load never produces a partly built index), and std::runtime_error
    /// if the file cannot be opened.
    static std::unique_ptr<FlatIndex> load(const std::string& path);

    /// Loads from a stream, which must be seekable (files and string streams are).
    static std::unique_ptr<FlatIndex> load(std::istream& in);

    std::size_t size() const {
        const auto guard = lock_.read();
        return live_;
    }
    std::size_t deleted_count() const { return 0; }
    std::size_t capacity() const {
        const auto guard = lock_.read();
        return vectors_.size();
    }
    std::size_t dim() const { return vectors_.dim(); }
    Metric metric() const { return metric_; }
    const MetadataStore& metadata() const { return metadata_; }

private:
    /// update() and upsert(): `metadata` null keeps the old metadata.
    bool replace(std::uint64_t id, std::span<const float> vector, const Metadata* metadata);  // caller holds the write lock
    void add_unlocked(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);  // caller holds the write lock
    bool remove_unlocked(std::uint64_t id);  // caller holds the write lock; never allocates
    void add_vector(std::span<const float> original);  // scratch_ (prepared) and its original; all-or-nothing
    std::span<const float> original(NodeId slot) const { return originals_ ? originals_->get(slot) : vectors_.get(slot); }
    std::vector<NodeId> matching_slots(const Filter& filter) const;  // caller holds a lock
    Record record(NodeId slot, bool with_vector, bool with_metadata) const;
    void add_batch_unlocked(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                            const Metadata* metadata);  // caller holds the write lock

    /// Compiles the options' filter (if any) into `holder` and returns the eligibility test.
    Eligibility eligibility(const SearchOptions& options, std::optional<CompiledFilter>& holder) const;

    /// Exact top-k over eligible vectors for one prepared query.
    std::vector<SearchResult> scan(const float* query, std::size_t k, const Eligibility& eligible,
                                   SearchStats* stats) const;

    Metric metric_;
    DistanceFn distance_;     // fastest kernel for this CPU, chosen once
    VectorStore vectors_;
    std::optional<VectorStore> originals_;  // cosine: the vectors as given, slot by slot
    IdMap ids_;
    MetadataStore metadata_;
    PreparedVector scratch_;  // reused by add(); searches use their own copy
    std::size_t live_ = 0;
    mutable SharedPool pool_;
    IndexLock lock_;  // reads share it, changes take it alone (see IndexLock)
};

}  // namespace vecdb