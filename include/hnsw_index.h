/**
 * @file hnsw_index.h
 * @brief Approximate nearest-neighbor search with a Hierarchical Navigable
 *        Small World (HNSW) graph.
 */
#pragma once
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include "distance.h"
#include "filter.h"
#include "health.h"
#include "metadata.h"
#include "prepared_vector.h"
#include "search_result.h"
#include "storage.h"
#include "codec.h"
#include "thread_pool.h"
#include "visited_list.h"

namespace vecdb {

/**
 * @brief SplitMix64 (Steele, Lea and Flood, 2014): a small, fast random number
 *        generator whose entire state is one 64-bit number.
 *
 * Used for random levels. Its state is saved as plain bytes, so a saved index
 * continues identically on every platform. (The text form that C++ streams
 * write for std::mt19937_64 differs between standard libraries: a file saved
 * with libstdc++ could not be loaded with libc++.)
 */
struct SplitMix64 {
    std::uint64_t state;
    explicit SplitMix64(std::uint64_t seed = 0) : state(seed) {}
    std::uint64_t operator()() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

/**
 * @brief Settings that control HNSW graph quality, memory and speed.
 */
struct HnswParams {
    /// Links per node on levels 1+ (2*M on level 0). Higher: better recall,
    /// more memory, slower inserts. Must be 2 to 2048; 8 to 48 recommended.
    std::size_t M = 16;
    /// How hard insertion searches for good neighbors. Higher: better graph,
    /// slower inserts. Must be at least 1.
    std::size_t ef_construction = 200;
    /// Seed for random levels, which are identical on every platform. Same seed
    /// and same operations give an identical graph on the same CPU kernel; the
    /// SIMD kernels (AVX-512, AVX2, NEON, scalar) round distances differently,
    /// so graphs can differ slightly between machines.
    std::uint64_t seed = 42;
    /// Repair the graph around a removed vector: each of its neighbors re-chooses
    /// its links from its own links plus the removed vector's links. Keeps search
    /// quality high under heavy churn; costs some time per remove.
    bool repair_on_remove = true;
    /// Compact storage of vectors (codecs), with re-ranking at full precision.
    QuantizationParams quantization;
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

    /// Adds many vectors at once. `vectors` holds ids.size() vectors back to
    /// back; `metadata`, if given, has one entry per vector. The index's lock is
    /// held throughout, so searches wait and then see all of the batch or none of it.
    ///
    /// **All or nothing.** Every vector, ID (against the index and within the
    /// batch) and metadata entry is checked before anything changes; any problem
    /// throws std::invalid_argument and leaves the index unchanged.
    ///
    /// **Threads.** Vectors that reuse removed slots are linked one at a time,
    /// exactly as add() would; the rest are linked into the graph on `threads`
    /// threads (0: all cores). With one thread the result is identical to calling
    /// add() for each vector in order; with more, the graph differs between runs
    /// (search quality does not).
    ///
    /// If memory runs out while linking, the batch's vectors are removed again:
    /// none of them remains, and every earlier vector is still stored and
    /// searchable. The graph is repaired around them when memory allows, as by
    /// remove(); some links may differ from before, as after any removal.
    /// `options` adds a deadline, cancellation and progress reports (see
    /// OperationOptions); a cancelled batch is undone and throws Cancelled.
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads = 0,
                   const OperationOptions& options = {});
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                   std::span<const Metadata> metadata, std::size_t threads = 0, const OperationOptions& options = {});

    /// The vector stored for `id`, exactly as it was given to add() or update()
    /// (for cosine indexes, before normalization: the original is kept for this),
    /// or nothing if `id` is not stored.
    /// "More like this": searches with the vector stored for `id`, under one lock
    /// (so it cannot change in between). Returns nothing if `id` is not stored.
    /// The record itself is left out unless `include_self` is true.
    /// A health report: storage, graph structure, an estimate of recall, and the
    /// removals and updates since the last upkeep (see health.h). Takes the shared
    /// lock, so searches continue meanwhile; then runs the sampled searches.
    HealthReport health(const HealthOptions& options = {}) const;

    /// Incremental upkeep, like a database's vacuum: consolidates links to removed
    /// vectors and reconnects unreachable ones (see UpkeepOptions). Takes the lock
    /// alone; use max_nodes to keep each call short, so searches run in between.
    /// A deadline or cancellation (options) stops it early without error: what was
    /// repaired stays repaired. When complete, the churn counters are reset.
    UpkeepStats upkeep(const UpkeepOptions& upkeep_options = {}, const OperationOptions& options = {});

    /// Automatic compact() once removed vectors' free slots exceed `fraction` of
    /// all slots (0: off, the default; at most 0.9). Runs inside the change that
    /// crosses the threshold, which then takes as long as a compact(); if it fails
    /// (out of memory) the change still stands and the next one tries again. A
    /// runtime setting: not saved with the index.
    void set_auto_compact(double fraction);
    double auto_compact() const;

    /// Rebuilds the index while it stays in use, like compact() but retraining the
    /// codec (RebuildOptions) and without blocking: a snapshot of the records is
    /// taken under the lock, a new index is built from it without any lock
    /// (searches and changes continue meanwhile), and then, under the lock, the
    /// IDs changed in between are copied in and the new index swapped in. Needs
    /// memory for a second copy while it runs. One rebuild at a time
    /// (std::logic_error otherwise). A deadline or cancellation throws Cancelled
    /// and leaves this index as it is; so does running out of memory.
    RebuildStats rebuild(const RebuildOptions& rebuild_options = {}, const OperationOptions& options = {});

    std::optional<std::vector<SearchResult>> search_by_id(std::uint64_t id, std::size_t k, std::size_t ef = 64,
                                                         const SearchOptions& options = {}, bool include_self = false) const;

    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;

    /// Trains the codec (HnswParams::quantization) on the stored vectors now, and
    /// encodes every vector; vectors added later are encoded as they arrive.
    /// Training again replaces the previous training. All or nothing. Throws
    /// std::logic_error if no codec is configured, or once only codes are kept
    /// (keep_full_vectors off: training again from decoded vectors could only lose
    /// precision, so the first training is final), std::invalid_argument if the
    /// index is empty. (Training also happens automatically: see train_after.)
    void train_codec();

    /// Whether searches use codes (a codec is configured and trained).
    bool codec_trained() const;

    /// Whether full-precision vector values are stored: false once an index that
    /// keeps only codes (a codec, keep_full_vectors off) has been trained.
    bool has_vector_values() const {
        const auto guard = lock_.read();
        return storage_.vectors().has_values();
    }

    /// The codec, for inspection while no other thread changes the index (like
    /// storage()); null without quantization.
    const VectorCodec* codec() const { return codec_.get(); }

    /// The code stored for `id`, or nothing if `id` is not stored or no codec
    /// is trained. For inspection and tests.
    std::optional<std::vector<std::uint8_t>> get_code(std::uint64_t id) const;

    /// Removes every listed ID that is stored and returns how many it removed;
    /// unknown and repeated IDs are skipped. Always completes, like remove_where.
    std::size_t remove_batch(std::span<const std::uint64_t> ids);

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
    /// **Always complete:** once the filter is accepted, every match is removed;
    /// running out of memory can only leave the graph less repaired around
    /// them (as remove() would), never a match behind.
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

    /// Deletes `id` for real: repairs the graph around it (if enabled), frees
    /// the ID for reuse and puts its slot on the free list. Returns false if the
    /// ID is not stored. If memory runs out during repair, the vector stays
    /// stored and the graph stays valid.
    bool remove(std::uint64_t id);

    /// Replaces the vector stored under `id`, keeping its metadata. Returns
    /// false if `id` is not stored. Throws std::invalid_argument on a wrong
    /// dimension, NaN or infinity (checked first, even for an unknown ID).
    ///
    /// The new version is built and linked in another slot while the old one
    /// stays live and findable; only then does the ID switch over and the old
    /// slot join the free list. A failure, including running out of memory,
    /// leaves the old vector exactly as it was. Capacity grows by at most one
    /// slot, which later inserts and updates reuse.
    bool update(std::uint64_t id, std::span<const float> vector);

    /// Replaces the vector and its whole metadata (fields not given are cleared;
    /// to change some fields only, use set_metadata()). Same guarantees.
    bool update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);

    /// Adds `id` if it is not stored, otherwise replaces its vector (keeping its
    /// metadata). Returns true if it was added, false if it was replaced.
    bool upsert(std::uint64_t id, std::span<const float> vector);

    /// Adds `id` with this metadata, or replaces its vector and whole metadata.
    /// Returns true if it was added, false if it was replaced.
    bool upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);

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
    void create_payload_index(std::string_view field) {
        const auto guard = lock_.write("create_payload_index");
        metadata_.index_field(field);
    }

    /// The index's planner settings (per-query settings in SearchOptions override them).
    void set_planner_params(const PlannerParams& p) {
        const auto guard = lock_.write("set_planner_params");
        validate(p);
        planner_ = p;
    }
    /// A copy, so it is safe to call while other threads change the settings.
    PlannerParams planner_params() const {
        const auto guard = lock_.read();
        return planner_;
    }
    const MetadataStore& metadata() const { return metadata_; }
    /// Visited lists kept for reuse by searches (released by compact()).
    std::size_t pooled_visited_lists() const { return visited_pool_.idle_count(); }

    /// Rebuilds the index from the live vectors only, with dense internal
    /// numbers and no free slots. User IDs are kept. The new index is built
    /// completely before replacing the old one, so if memory runs out the old
    /// index is left intact.
    /// `options`: a deadline, cancellation and progress reports; a cancelled
    /// compaction leaves the index untouched and throws Cancelled.
    CompactStats compact(const OperationOptions& options = {});

    /// Number of live vectors.
    /// Saves the whole index (vectors, graph, metadata, settings) to `path`. It is
    /// written to a temporary file next to it ("<path>.tmp-" and a unique
    /// suffix) and then renamed into place, so a failure never damages an
    /// existing file at `path`, and saves to one path at the same time each
    /// produce a complete file (the last one wins). Searches may run during a save;
    /// changes attempted during one throw std::logic_error. Throws
    /// std::runtime_error if the file cannot be written.
    void save(const std::string& path, const OperationOptions& options = {}) const;

    /// Saves to a stream (for example a std::ostringstream).
    /// With `options`: a deadline, cancellation and progress reports; a cancelled
    /// save throws Cancelled, leaving any existing file untouched and no temporary
    /// file (a stream may hold an incomplete index).
    void save(std::ostream& out, const OperationOptions& options = {}) const;

    /// Loads an index saved by save(). Throws FormatError if the file is damaged,
    /// truncated, from an unsupported version or of the other index kind (a
    /// failed load never produces a partly built index), and std::runtime_error
    /// if the file cannot be opened.
    static std::unique_ptr<HnswIndex> load(const std::string& path);

    /// Loads from a stream, which must be seekable (files and string streams are).
    static std::unique_ptr<HnswIndex> load(std::istream& in);

    std::size_t size() const {
        const auto guard = lock_.read();
        return live_;
    }
    /// Slots of removed vectors still occupying memory (waiting to be reused).
    std::size_t deleted_count() const {
        const auto guard = lock_.read();
        return storage_.size() - live_;
    }
    /// All slots, live or removed.
    std::size_t capacity() const {
        const auto guard = lock_.read();
        return storage_.size();
    }

    /// The dimension of vectors and queries as given.
    std::size_t dim() const { return input_dim_; }
    /// The dimensions stored and searched: dim(), or fewer with Matryoshka truncation.
    std::size_t indexed_dim() const { return storage_.vectors().dim(); }
    Metric metric() const { return metric_; }
    const HnswParams& params() const { return params_; }

    /// Highest level of a live node, or -1 when the index has no live vectors.
    int max_level() const {
        const auto guard = lock_.read();
        return max_level_;
    }
    /// Node where every search starts, or kEmpty when there are no live vectors.
    NodeId entry_point() const {
        const auto guard = lock_.read();
        return entry_;
    }
    /// Read-only access to the underlying storage (used by tests to check the graph).
    const Storage& storage() const { return storage_; }

private:
    /// A stored vector as input again (for compact and search_by_id): the vector as
    /// given if kept, otherwise the stored one, padded with zeros to dim() when
    /// truncating (truncation ignores those values).
    std::vector<float> as_input(NodeId slot) const;

    /// Whether only codes are kept (no stored floats): after training, when
    /// keep_full_vectors is false and a codec is configured.
    bool codes_only() const { return !storage_.vectors().has_values(); }
    /// The stored vector of `slot`, padded to the stride: the floats, or decoded
    /// from its code into `buffer` (codes only).
    std::span<const float> stored_padded(NodeId slot, std::vector<float>& buffer) const;
    /// The stored vector of `slot` (indexed_dim() values), as a copy.
    std::vector<float> stored_copy(NodeId slot) const;
    /// A vector to read back for `slot`: the vector as given if kept, otherwise
    /// the stored one (decoded, with codes only).
    std::vector<float> read_back(NodeId slot) const;

    /// Draws a random top level: floor(-ln(u) / ln(M)), u uniform in (0, 1].
    int random_level();

    /// Distance from a prepared vector (padded row) to stored node `node`.
    float distance_to(const float* query, NodeId node) const;

    /// Codes: make room for `slot`'s code (may throw: called before any change),
    /// encode the vector stored in `slot` (cannot fail), train when due.
    void reserve_code(NodeId slot);
    void encode_slot(NodeId slot);
    void maybe_train() noexcept;
    void train_codec_unlocked();

    /// Distance between two stored nodes.
    float distance_between(NodeId a, NodeId b) const;

    /// True if two stored nodes hold bit-identical vectors.
    bool same_vector(NodeId a, NodeId b) const;

    /// Locks for linking many vectors in parallel (add_batch): each node's link
    /// lists are guarded by one of a fixed array of mutexes (lock striping, as in
    /// hnswlib), and the entry point and top level by `top`.
    struct LinkLocks {
        explicit LinkLocks(std::size_t stripes) : lists(stripes) {}
        std::mutex& of(NodeId n) { return lists[n & (lists.size() - 1)]; }
        std::vector<std::mutex> lists;  // size: a power of two
        std::mutex top;
    };
    static constexpr std::size_t kLinkStripes = std::size_t(1) << 16;

    /// Greedy walk on one level: moves to any closer neighbor until none is
    /// closer. Never steps onto `exclude` or onto a node below `level`. With
    /// `locks`, each link list is read under its lock (parallel linking).
    Candidate greedy_closest(const float* query, Candidate start, int level, NodeId exclude,
                             LinkLocks* locks = nullptr) const;

    /// Beam search on one level. Returns up to `ef` closest eligible nodes,
    /// closest first. Ineligible nodes (removed, or failing the filter) are
    /// traversed but never returned. `exclude` (a node being inserted) is never
    /// visited. Counts work in `stats` if given.
    std::vector<Candidate> search_level(const float* query, const std::vector<Candidate>& entries,
                                        std::size_t ef, int level, VisitedList& visited,
                                        NodeId exclude, const Eligibility& eligible,
                                        SearchStats* stats = nullptr, LinkLocks* locks = nullptr) const;

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
    void link_new_node(NodeId node, int level, NodeId ignore = kEmpty);

    /// Links `node` on its levels, starting from `entry` on level `top`, reading
    /// the vector from `query`. With `locks`, every link list is read and written
    /// under its lock. Does not update the entry point.
    void link_levels(NodeId node, int level, NodeId ignore, const float* query, NodeId entry, int top,
                     LinkLocks* locks);

    /// add_batch's work; the caller holds the write lock. `metadata` is an array
    /// of ids.size() entries, or null.
    void add_batch_unlocked(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                            const Metadata* metadata, std::size_t threads, const OperationOptions& options);

    /// Links one stored, not yet linked node in parallel with others (add_batch).
    void link_concurrently(NodeId node, int level, LinkLocks& locks);

    /// Records for `slots` (in the given order), with vectors and metadata as requested.
    Record record(NodeId slot, bool with_vector, bool with_metadata) const;

    /// Removes the given live slots (no repeats), then repairs around them, best
    /// effort; the caller holds the write lock. Always completes.
    std::size_t remove_slots(const std::vector<NodeId>& slots);

    /// Live slots matching `filter`; the caller holds a lock.
    std::vector<NodeId> matching_slots(const Filter& filter) const;

    /// Undoes a failed batch: removes every node it stored (newest first), then
    /// repairs the graph around them if memory allows.
    void roll_back_batch(std::span<const std::uint64_t> ids, const std::vector<NodeId>& nodes,
                         const std::vector<MetadataStore::WriteUndo>& undo) noexcept;

    /// update() and upsert(): `metadata` null keeps the old metadata.
    bool replace(std::uint64_t id, std::span<const float> vector, const Metadata* metadata);  // caller holds the write lock
    /// The caller holds the write lock. `prepared`: `vector` is a stored vector of
    /// this index (indexed_dim() values, already normalized), copied exactly; for
    /// compact() without originals, so rebuilding never changes a stored vector.
    void add_unlocked(std::uint64_t id, std::span<const float> vector, const Metadata& metadata, bool prepared = false);

    /// Sets `node`'s links on `level` and adds the reverse link to each neighbor,
    /// re-selecting a neighbor's list with the heuristic when it is full.
    void connect(NodeId node, const std::vector<NodeId>& neighbors, int level, LinkLocks* locks = nullptr);
    /// The two halves of connect(): set the node's own list, then add the reverse
    /// links (shrinking full lists). Parallel linking does all of the first before
    /// any of the second.
    void set_own_links(NodeId node, const std::vector<NodeId>& neighbors, int level, LinkLocks* locks);
    void link_back(NodeId node, const std::vector<NodeId>& neighbors, int level, LinkLocks* locks);

    /// Before `node` is removed: each live neighbor re-chooses its links from
    /// its own links plus `node`'s links, minus `node`, on every level.
    void repair_neighbors(NodeId node);

    /// Makes the live node with the highest level the entry point, or sets the
    /// entry point to kEmpty and the top level to -1 if no node is live.
    void choose_new_entry();

    std::size_t input_dim_;  // as given; the stored vectors may be truncated (see indexed_dim)
    Metric metric_;
    DistanceFn distance_;  // fastest kernel for this CPU, chosen once
    HnswParams params_;
    double level_factor_;  // 1 / ln(M)
    Storage storage_;
    std::size_t stride_;
    PreparedVector scratch_;  // reused by add(); search() uses its own copy
    SplitMix64 rng_;  // random levels
    std::unique_ptr<VectorCodec> codec_;  // null without quantization
    std::vector<std::uint8_t> codes_;     // trained: code_size() bytes per slot (at least as many slots as stored)
    NodeId entry_ = kEmpty;
    int max_level_ = -1;
    std::size_t live_ = 0;
    std::vector<NodeId> free_slots_;  // slots of removed vectors, reused last in first out
    std::size_t removals_since_upkeep_ = 0;  // for health(): counted where each change is final
    std::size_t updates_since_upkeep_ = 0;
    double auto_compact_ = 0;  // see set_auto_compact
    std::optional<std::unordered_set<std::uint64_t>> changed_;  // while rebuild() runs: IDs changed since its snapshot
    void note_change(std::uint64_t id) {  // caller holds the write lock; call before changing anything
        if (changed_) changed_->insert(id);
    }
    CompactStats compact_unlocked(const OperationOptions& options);  // caller holds the write lock
    void maybe_auto_compact() noexcept;                             // caller holds the write lock
    UpkeepStats upkeep_unlocked(const UpkeepOptions& upkeep_options, const OperationOptions& options);  // likewise
    MetadataStore metadata_;
    PlannerParams planner_;
    mutable VisitedListPool visited_pool_;
    mutable SharedPool pool_;
    IndexLock lock_;  // reads share it, changes take it alone (see IndexLock)
};

}  // namespace vecdb