/**
 * @file collection.h
 * @brief One object for documents: a dense vector, text and a sparse vector,
 *        each optional, with metadata, under one ID space; hybrid search by
 *        default; durable as a whole.
 *
 * A Collection keeps up to three sides, each holding the documents that have
 * that part:
 *  - dense:  an HnswIndex of the documents' vectors (options.dim > 0);
 *  - text:   a SparseIndex of BM25 weights of the documents' text (options.text);
 *  - sparse: a SparseIndex of learned sparse vectors such as SPLADE (options.sparse).
 * Text and learned sparse vectors are kept apart because their term numbers
 * mean different things. Metadata, text and the ID table live in the
 * collection itself, so filters reach every side.
 *
 * A change is checked in full before anything is touched, logged once (when
 * durable), and applied to every side; if a later side runs out of memory the
 * earlier ones are restored. If even that fails, the collection refuses further
 * changes (searches continue) until it is reopened.
 *
 * search() takes any combination of a vector, text and a sparse vector: one
 * side gives that side's ranking, several are fused by Reciprocal Rank Fusion
 * (or weighted scores).
 */
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <thread>
#include <string>
#include <unordered_map>
#include <vector>

#include "change_log.h"
#include "durable.h"
#include "hnsw_index.h"
#include "sparse.h"

namespace vecdb {

/// How maintenance runs: only when maintain() is called, or also on a background thread.
enum class Maintenance { Manual, Background };

/// When maintenance work is due. Churn is dense vectors replaced or removed
/// since that work last ran, over the live ones (additions do not damage the graph).
struct MaintenancePolicy {
    Maintenance mode = Maintenance::Manual;
    std::chrono::milliseconds interval{1000};  ///< Background: how often to look for due work
    std::chrono::milliseconds budget{250};     ///< Background: time per round for health checks and upkeep
    // Checkpoints (durable collections): the log since the last snapshot.
    std::uint64_t checkpoint_bytes = 64ull << 20;
    std::uint64_t checkpoint_records = 100000;
    std::chrono::seconds checkpoint_age{600};  ///< and any record waiting longer than this
    // The dense side.
    double health_check_churn = 0.05;  ///< check reachability past this; upkeep if anything is wrong
    double upkeep_churn = 0.2;         ///< upkeep past this, whatever the health
    std::size_t upkeep_nodes = 0;      ///< nodes per upkeep call (0: no limit; the budget still applies)
    double rebuild_check_churn = 0.5;  ///< with a codec: check its drift past this
    double rebuild_drift = 4.0;        ///< rebuild when its error exceeds this many times a retrained one's
};

/// What a maintenance round did, and why.
struct MaintenanceReport {
    std::chrono::system_clock::time_point when{};
    std::chrono::microseconds elapsed{0};
    std::optional<HealthReport> health;  ///< if a health check ran
    bool upkept = false;
    UpkeepStats upkeep;
    bool rebuilt = false;
    RebuildStats rebuild;
    bool checkpointed = false;
    std::uint64_t checkpoint_lsn = 0;
    bool budget_exhausted = false;   ///< due work was left for the next round
    std::vector<std::string> notes;  ///< each decision, in words
    std::string error;               ///< background rounds: what stopped this one
};

/// The counters maintenance decides from.
struct MaintenanceStatus {
    std::uint64_t log_bytes = 0;       ///< log since the last snapshot (durable collections)
    std::uint64_t log_records = 0;
    std::chrono::seconds since_checkpoint{0};
    double churn_since_upkeep = 0;
    double churn_since_rebuild = 0;
    std::size_t rounds = 0;            ///< maintenance rounds run
};

/// Ready-made settings (CollectionOptions::preset).
enum class Preset {
    Balanced,    ///< M 16, ef_construction 200 (24 and 300 from a million vectors); full precision; search beam 128
    HighRecall,  ///< M 32, ef_construction 400 (48 and 500 from a million); full precision; search beam 256
    LowMemory,   ///< M 12, ef_construction 150; SQ8 codes only (vectors about 4x smaller); search beam 128
};

struct CollectionOptions {
    std::size_t dim = 0;             ///< dense vector length; 0: no dense side
    Metric metric = Metric::Cosine;
    HnswParams hnsw;
    Schema schema = Schema::dynamic();
    bool text = true;                ///< a BM25 side for documents' text
    Bm25Params bm25;                 ///< avg_doc_length: set it near your corpus' mean token count
    bool sparse = false;             ///< a side for learned sparse vectors (SPLADE)
    bool store_text = true;          ///< keep the text, for get() and search results
    DurableOptions durability;       ///< for open()
    MaintenancePolicy maintenance;   ///< not stored: load() starts with Manual
    std::size_t ef = 0;              ///< default dense search beam (0: max(candidates, 64)); not stored

    /// Settings for `dim`-dimensional vectors (0: text or sparse only) and about
    /// `expected_size` documents: graph, compression and search beam as the preset
    /// says, cosine distance, BM25 text, and background maintenance. Adjust any
    /// field afterwards. Throws std::invalid_argument for an unknown preset.
    static CollectionOptions preset(std::size_t dim, Preset preset = Preset::Balanced, std::size_t expected_size = 100000);
};

/// A document. Every part is optional; a part the collection was not created
/// for is an error.
struct Document {
    std::vector<float> vector;
    std::string text;
    SparseVector sparse;
    Metadata metadata;
};

/// A document read back. text is empty unless store_text.
struct DocumentRecord {
    std::uint64_t id = 0;
    std::optional<std::vector<float>> vector;
    std::string text;
    std::optional<SparseVector> sparse;
    Metadata metadata;
};

struct CollectionQuery {
    std::vector<float> vector;  ///< dense query (empty: none)
    std::string text;           ///< BM25 query (empty: none)
    SparseVector sparse;        ///< learned sparse query (empty: none)
};

struct CollectionSearchOptions {
    std::optional<Filter> filter;                    ///< on the collection's metadata, for every side
    std::function<bool(std::uint64_t)> predicate;    ///< must be thread-safe if used concurrently
    Fusion fusion = Fusion::Rrf;
    double rrf_k = 60.0;
    double vector_weight = 1.0, text_weight = 1.0, sparse_weight = 1.0;
    std::size_t candidates = 0;  ///< per side; 0: max(4 * k, 20)
    std::size_t ef = 0;          ///< dense beam width; 0: max(candidates, 64)
    bool with_metadata = false;
    bool with_text = false;
    /// Stop every side when this moment passes, or when another thread sets the
    /// flag: the hits found so far are returned (fused and sorted), and
    /// *stopped_early, if given, is set to true.
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    bool* stopped_early = nullptr;
};

struct CollectionHit {
    std::uint64_t id = 0;
    double score = 0;  ///< fused, higher is better
    std::optional<std::size_t> vector_rank, text_rank, sparse_rank;  ///< 1-based
    std::optional<float> vector_distance;
    std::optional<float> text_score, sparse_score;  ///< dot products
    std::optional<Metadata> metadata;
    std::optional<std::string> text;
};

class Collection {
public:
    /// An in-memory collection. Throws std::invalid_argument for invalid options
    /// (no side at all, invalid HNSW or BM25 settings).
    explicit Collection(CollectionOptions options);
    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;
    ~Collection();

    /// A durable collection in `dir`: created, or recovered (see durable.h).
    /// When recovering, dim, metric and the sides must match the stored ones
    /// (std::invalid_argument); other options come from the directory.
    static std::unique_ptr<Collection> open(const std::string& dir, CollectionOptions options, RecoveryStats* stats = nullptr);
    /// A snapshot and a new log segment (durable collections only: std::logic_error).
    std::uint64_t checkpoint();
    bool durable() const { return durable_ != nullptr; }

    /// Adds a document (std::invalid_argument for a duplicate ID or invalid parts).
    void add(std::uint64_t id, const Document& document);
    /// Adds or replaces the whole document; returns true if added.
    bool upsert(std::uint64_t id, const Document& document);
    bool remove(std::uint64_t id);
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::size_t remove_where(const Filter& filter);
    /// Changes only the given fields (unset() clears one). False if unknown.
    bool set_metadata(std::uint64_t id, const Metadata& metadata);
    std::size_t set_metadata_where(const Filter& filter, const Metadata& changes);

    bool contains(std::uint64_t id) const;
    std::size_t size() const;
    std::size_t count_where(const Filter& filter) const;
    std::optional<DocumentRecord> get(std::uint64_t id) const;
    const CollectionOptions& options() const { return options_; }

    /// The k best documents for the query (at least one part; std::invalid_argument
    /// otherwise, or for a part the collection does not have).
    std::vector<CollectionHit> search(const CollectionQuery& query, std::size_t k, const CollectionSearchOptions& options = {}) const;

    void save(const std::string& path) const;
    void save(std::ostream& out) const;
    static std::unique_ptr<Collection> load(const std::string& path);
    static std::unique_ptr<Collection> load(std::istream& in);

    /// Applies logged records, as recovery does; returns true if an unfinished
    /// final operation was dropped (see HnswIndex::replay).
    bool replay(std::span<const WalRecord> records);

    /// Runs the maintenance that is due (see MaintenancePolicy): a health check and
    /// upkeep of the dense side within `budget` (0: no limit), a rebuild if its
    /// codec drifted (once started, a rebuild runs to the end), then a checkpoint if
    /// the log has grown. Searches and changes continue meanwhile. Maintenance is not
    /// logged: the next snapshot keeps its effect. Refused from inside a search
    /// (std::logic_error).
    MaintenanceReport maintain(std::chrono::milliseconds budget = std::chrono::milliseconds(0));
    /// Changes the policy; starts or stops the background thread as needed.
    /// Throws std::invalid_argument for intervals under 1 ms or negative thresholds.
    void set_maintenance(const MaintenancePolicy& policy);
    std::optional<MaintenanceReport> last_maintenance() const;
    MaintenanceStatus maintenance_status() const;

    /// Read-only access to the sides (null if absent), e.g. for health reports.
    const HnswIndex* dense_index() const { return dense_.get(); }
    const SparseIndex* text_index() const { return text_.get(); }
    const SparseIndex* sparse_index() const { return sparse_.get(); }

private:
    struct Prepared;  // a checked document (its metadata is checked under the lock)
    Prepared prepare(const Document& d) const;
    bool apply_upsert(std::uint64_t id, const Prepared& p);
    bool apply_remove(std::uint64_t id);
    void check_changeable() const;
    bool searching_here() const;  // inside a search of this collection, on this thread
    std::vector<NodeId> matching(const Filter& filter) const;
    void save_unlocked(std::ostream& out) const;

    CollectionOptions options_;
    Bm25 bm25_;
    std::unique_ptr<HnswIndex> dense_;
    std::unique_ptr<SparseIndex> text_, sparse_;
    MetadataStore metadata_;
    std::unordered_map<std::uint64_t, NodeId> slot_of_;
    std::vector<std::uint64_t> id_of_;
    std::vector<std::uint8_t> live_;
    std::vector<std::string> texts_;
    std::vector<NodeId> free_;
    bool failed_ = false;  // a rollback failed: changes refused
    mutable std::shared_mutex mutex_;
    ChangeLog log_;
    std::shared_ptr<detail::DurableState> durable_;
    // Maintenance.
    void stop_maintenance();
    std::atomic<std::uint64_t> changed_since_upkeep_{0}, changed_since_rebuild_{0};
    std::atomic<std::uint64_t> checkpoint_lsn_{1};      // the last snapshot's LSN
    std::atomic<std::int64_t> checkpoint_time_{0};      // and when it was taken (steady clock, ns)
    std::mutex maintenance_mutex_;                      // one round at a time
    mutable std::mutex report_mutex_;
    std::optional<MaintenanceReport> last_report_;
    std::size_t rounds_ = 0;
    std::thread maintainer_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
};

}  // namespace vecdb
