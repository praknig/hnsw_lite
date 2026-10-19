/**
 * @file disk_index.h
 * @brief A disk-based index (DiskANN-style): the graph and full vectors stay on
 *        disk; only compact PQ codes, IDs and metadata are held in memory.
 *
 * File layout (all numbers little-endian), in 4 KB blocks:
 *   block 0       header: sizes, the medoid, section offsets and checksums, and
 *                 a checksum of the header itself
 *   blocks 1..    node records: the full (prepared) vector, the degree, and R
 *                 neighbor slots. Several records share a block when they fit;
 *                 a larger record takes whole blocks of its own. No record
 *                 crosses a block boundary, so one read fetches one node.
 *   then          block checksums (one per node block), the PQ codebooks and
 *                 codes, the user IDs, and the metadata: loaded at open().
 * Every in-memory section is verified at open(); every node block is verified
 * when read, so damage is reported (FormatError) where it is met.
 *
 * Read-only: built once by build(), then opened and searched; changes require
 * a rebuild. Building holds the vectors in memory; searching does not.
 */
#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "codec.h"
#include "filter.h"
#include "health.h"
#include "metadata.h"
#include "vamana.h"

namespace vecdb {

/// Positional reads from a read-only file, safe from many threads at once
/// (pread on POSIX, ReadFile with an offset on Windows). On every platform the
/// file may be deleted or replaced while open; reads continue from the old one.
class BlockFile {
public:
    explicit BlockFile(const std::string& path);  ///< throws std::runtime_error if it cannot be opened
    ~BlockFile();
    BlockFile(const BlockFile&) = delete;
    BlockFile& operator=(const BlockFile&) = delete;
    std::uint64_t size() const { return size_; }
    /// Reads exactly n bytes at `offset`. Throws FormatError past the end of the
    /// file, std::runtime_error for other read errors.
    void read(std::uint64_t offset, void* buffer, std::size_t n) const;
    /// Delays every read by `latency`, like slow storage: for tests and benchmarks.
    void set_simulated_latency(std::chrono::microseconds latency) { latency_ = latency; }

private:
    std::intptr_t handle_ = -1;
    std::uint64_t size_ = 0;
    std::chrono::microseconds latency_{0};
};

/// How an index file is opened.
struct DiskOpenOptions {
    /// Threads that read node blocks concurrently: the W reads of a search round
    /// overlap instead of following one another (and several searches share the
    /// threads). 0: reads on the searching thread, one after another.
    std::size_t io_threads = 0;
    /// Delays every read, like slow storage (for tests and benchmarks; 0: none).
    std::chrono::microseconds simulated_latency{0};
};

namespace detail {
class IoPool;  // disk_index.cpp
}

/// How a disk search explores: a list of L candidates (ranked by PQ distance),
/// of which the W best unread ones are read from disk each round.
struct DiskSearchParams {
    std::size_t L = 100;           ///< candidate list size (at least k is used): recall against reads
    std::size_t beam_width = 4;    ///< nodes read per round (W): fewer rounds against more reads
};

/// What a disk search did.
struct DiskSearchStats {
    std::size_t nodes_read = 0;    ///< nodes read from disk (each one block, or more for large records)
    std::size_t blocks_read = 0;   ///< 4 KB blocks read
    std::size_t rounds = 0;        ///< read rounds (round trips; reads within one could overlap)
    bool stopped_early = false;    ///< a deadline or cancel flag stopped it
};

struct DiskIndexParams {
    VamanaParams graph;            ///< the graph (R, L, alpha, seed)
    /// PQ parts of the in-memory codes that guide searches (0: one per 2
    /// dimensions). Finer than PQ's general default: here the codes choose which
    /// nodes to read, and coarse codes waste reads (measured: 32 dimensions with 4
    /// parts reached recall@10 0.71 at L 50, with 16 parts 1.00, for the same reads).
    std::size_t pq_subspaces = 0;
};

/// Building from a vectors file that need not fit in memory (DiskIndex::build_from_file).
struct LargeBuildParams {
    DiskIndexParams index;
    /// Memory for vectors and graphs while building one partition (bytes). The
    /// PQ codes, IDs and metadata (what a search holds in memory) come on top.
    std::size_t memory_budget = std::size_t(1) << 30;
    /// Partitions each vector joins: at least 2 when there are several, so that
    /// neighboring partitions share vectors and their graphs connect.
    std::size_t replicas = 2;
    /// 0: automatic (1 if everything fits; otherwise enough that no partition can
    /// exceed the budget, with two thirds of slack). Explicit values below
    /// ceil(n * replicas / per-partition capacity) + replicas are rejected.
    std::size_t partitions = 0;
};

struct LargeBuildStats {
    std::size_t partitions = 0;          ///< partitions built (1: built in memory)
    std::size_t largest_partition = 0;   ///< vectors in the largest (never beyond the budget: full partitions are skipped)
    std::size_t merged_and_pruned = 0;   ///< vectors whose joined lists exceeded R and were pruned again
};

/// Writes vectors (n * dim values) as a vectors file for build_from_file: the
/// .fbin layout (a 32-bit count, a 32-bit dimension, then the floats, all
/// little-endian), as used by DiskANN and the big-ANN benchmarks.
void write_vectors_file(const std::string& path, std::span<const float> vectors, std::size_t dim);

class DiskIndex {
public:
    static constexpr std::size_t kBlock = 4096;

    /// Builds an index over `vectors` (n * dim values) with user IDs `ids` (and
    /// optionally one Metadata per vector), and writes it to `path`, replacing any
    /// file there only once complete. L2 and cosine. Throws std::invalid_argument
    /// for invalid input (sizes, repeated IDs, non-finite values, settings).
    static void build(const std::string& path, std::span<const std::uint64_t> ids, std::span<const float> vectors,
                      std::size_t dim, Metric metric, const DiskIndexParams& params = {},
                      std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic());

    /// Builds from a vectors file (see write_vectors_file) that need not fit in
    /// memory (as DiskANN does): if it fits the budget it is built in memory;
    /// otherwise k-means on a sample divides it into partitions that fit, each
    /// vector joins its `replicas` nearest, a Vamana graph is built per partition
    /// (one at a time), and the graphs are merged in one pass, lists beyond R
    /// pruned again using the PQ codes. Temporary files go to a directory next to
    /// `path`, removed afterwards. IDs default to 0..n-1. Throws
    /// std::invalid_argument for an invalid file or settings.
    static void build_from_file(const std::string& path, const std::string& vectors_path, Metric metric,
                                const LargeBuildParams& params = {}, std::span<const std::uint64_t> ids = {},
                                std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic(),
                                LargeBuildStats* stats = nullptr);

    /// Opens an index file, verifying everything held in memory. Throws
    /// FormatError for a damaged or invalid file.
    static std::unique_ptr<DiskIndex> open(const std::string& path, const DiskOpenOptions& options = {});
    ~DiskIndex();

    std::size_t size() const { return ids_.size(); }
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    std::size_t max_degree() const { return R_; }
    NodeId medoid() const { return medoid_; }
    const MetadataStore& metadata() const { return metadata_; }
    const PQCodec& codec() const { return *codec_; }

    std::uint64_t id_of(NodeId node) const { return ids_.at(node); }
    std::optional<NodeId> node_of(std::uint64_t id) const;

    /// One node as stored on disk: its (prepared) vector and its neighbors.
    struct Node {
        std::vector<float> vector;
        std::vector<NodeId> neighbors;
    };
    /// Reads a node from disk, verifying its block(s). Throws FormatError for
    /// damage, std::out_of_range for an unknown node.
    Node read_node(NodeId node) const;

    /// The k nearest vectors to `query` (ids are user IDs, distances exact).
    /// Candidates are ranked by PQ distance in memory; the best unread ones are
    /// read from disk, W per round, their exact distances computed from the full
    /// vectors read, and their neighbors added; the search ends when the best L
    /// candidates were all read. Deadlines and cancellation (SearchOptions) are
    /// checked every round; they stop it with the best results so far.
    ///
    /// Filters and predicates: only matching vectors are returned, and never
    /// fewer than k if k match (the search widens L until they are found). A
    /// filter (without a predicate) matching at most max(L, k) vectors skips the
    /// graph: those vectors are read directly, an exact answer. Safe to call from
    /// many threads (predicates must be too). Throws FormatError if a node block
    /// read is damaged, std::invalid_argument for an unknown field under a strict schema.
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, const DiskSearchParams& params = {},
                                     const SearchOptions& options = {}, DiskSearchStats* stats = nullptr) const;

    /// Searches several queries (n * dim values) in parallel (`threads`, 0: all
    /// cores); one result list per query.
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k,
                                                        const DiskSearchParams& params = {}, std::size_t threads = 0,
                                                        const SearchOptions& options = {}) const;

    /// "More like this": searches with the vector stored for `id`, leaving the
    /// record itself out unless `include_self`. Nothing if `id` is not stored.
    std::optional<std::vector<SearchResult>> search_by_id(std::uint64_t id, std::size_t k, const DiskSearchParams& params = {},
                                                          const SearchOptions& options = {}, bool include_self = false) const;

    /// Records, read from disk on demand. Vectors are returned as stored, that is
    /// normalized for cosine (the disk index keeps no copy as given).
    bool contains(std::uint64_t id) const { return node_of_.count(id) > 0; }
    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;
    std::optional<Record> get(std::uint64_t id) const;
    /// Pages of records in ascending ID order, as for the in-memory indexes.
    Page scan(const ScanOptions& options = {}) const;

    /// A health report (see health.h): degrees, reachability from the medoid, an
    /// estimate of recall, and PQ drift. Reads every node once, in order, keeping
    /// the neighbor lists in memory (4R bytes per vector); the exact answers for
    /// the recall sample are computed during that pass. The index is read-only,
    /// so churn is always 0; underfull means fewer than R/2 neighbors.
    HealthReport health(const HealthOptions& options = {}) const;

    /// Blocks read from disk since opening (each node read counts its blocks).
    std::uint64_t blocks_read() const { return blocks_read_.load(std::memory_order_relaxed); }

private:
    DiskIndex() = default;
    std::uint64_t node_offset(NodeId node) const;  // file offset of the node's record
    std::uint64_t node_block(NodeId node) const;   // its first node block

    std::unique_ptr<BlockFile> file_;
    std::unique_ptr<detail::IoPool> io_;  // after file_: destroyed first, as its reads use file_
    std::size_t dim_ = 0, R_ = 0, node_bytes_ = 0, nodes_per_block_ = 0, blocks_per_node_ = 0;
    Metric metric_ = Metric::L2;
    DistanceFn distance_ = nullptr;
    NodeId medoid_ = 0;
    std::vector<std::uint32_t> block_crc_;
    std::unique_ptr<PQCodec> codec_;
    std::vector<std::uint8_t> codes_;
    std::vector<std::uint64_t> ids_;
    std::unordered_map<std::uint64_t, NodeId> node_of_;
    std::vector<std::pair<std::uint64_t, NodeId>> by_id_;  // sorted by ID, for scan()
    MetadataStore metadata_;
    mutable std::atomic<std::uint64_t> blocks_read_{0};
};

}  // namespace vecdb