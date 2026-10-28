/**
 * @file sparse.h
 * @brief Sparse vectors (SPLADE, BM25 weights), an inverted index for them, a
 *        BM25 encoder, and hybrid search fusing dense and sparse results.
 *
 * A sparse vector is a set of (term, weight) pairs; similarity is the dot
 * product over shared terms. SparseIndex keeps one posting list per term and
 * answers queries exactly: it accumulates the products of the query's terms
 * over their posting lists, then keeps the k best. Search cost is the total
 * length of the query terms' posting lists, not the size of the index.
 *
 * Results use the library's convention, lower is better: distance = -score.
 *
 * hybrid_search() runs a dense query on any dense index (HnswIndex, FlatIndex)
 * and a sparse query on a SparseIndex holding the same IDs, and fuses the two
 * ranked lists by ID: Reciprocal Rank Fusion (default) or normalized weighted
 * scores.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "change_log.h"
#include "sparse_vector.h"
#include "durable.h"
#include "filter.h"
#include "metadata.h"
#include "search_result.h"
#include "thread_pool.h"

namespace vecdb {


/// Dot product of two sparse vectors (any order).
double sparse_dot(const SparseVector& a, const SparseVector& b);

/// A record read back from a SparseIndex.
struct SparseRecord {
    std::uint64_t id = 0;
    SparseVector vector;  ///< sorted by term
    Metadata metadata;
};

class SparseIndex {
public:
    explicit SparseIndex(Schema schema = Schema::dynamic());

    /// Throws std::invalid_argument for a duplicate ID, mismatched lengths,
    /// repeated terms, non-finite weights, or invalid metadata (the index is
    /// then unchanged).
    void add(std::uint64_t id, const SparseVector& vector, const Metadata& metadata = {});
    bool remove(std::uint64_t id);
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::size_t remove_where(const Filter& filter);
    /// Replaces the vector (metadata kept), or vector and metadata. False if unknown.
    bool update(std::uint64_t id, const SparseVector& vector);
    bool update(std::uint64_t id, const SparseVector& vector, const Metadata& metadata);
    /// Adds or replaces; returns true if added.
    bool upsert(std::uint64_t id, const SparseVector& vector);
    bool upsert(std::uint64_t id, const SparseVector& vector, const Metadata& metadata);
    /// Changes only the given fields (unset() clears one). False if unknown.
    bool set_metadata(std::uint64_t id, const Metadata& metadata);

    bool contains(std::uint64_t id) const;
    std::size_t size() const;
    std::optional<SparseVector> get_vector(std::uint64_t id) const;
    std::optional<Metadata> get_metadata(std::uint64_t id) const;
    /// Every record, in ID order (for exports and tests).
    std::vector<SparseRecord> records() const;

    /// The k documents with the highest dot product with `query`, best first
    /// (distance = -score; ties by ID). Only documents sharing a term with the
    /// query are candidates. Honors options.filter, options.predicate,
    /// options.deadline and options.cancel (best results so far) and
    /// options.stats; other options do not apply. Throws std::invalid_argument
    /// for an invalid query.
    std::vector<SearchResult> search(const SparseVector& query, std::size_t k, const SearchOptions& options = {}) const;

    /// Live documents containing `term` (for IDF weights, as Bm25 uses).
    std::size_t document_frequency(std::uint32_t term) const;
    /// Distinct terms in live documents.
    std::size_t terms() const;
    /// Posting entries, live and awaiting compaction.
    std::size_t postings() const;
    std::size_t dead_postings() const;
    /// Document slots, live and awaiting compaction.
    std::size_t slots() const;
    /// The metadata schema in force (for a dynamic schema, the fields seen so far).
    Schema schema() const;
    /// The metadata fields with a payload index (none for a sparse index today).
    std::vector<std::string> payload_indexes() const;

    /// Drops removed documents from posting lists and renumbers slots. Runs by
    /// itself once dead entries outnumber live ones; explicit calls are optional.
    void compact();

    void save(const std::string& path) const;
    void save(std::ostream& out) const;
    /// Throws FormatError for a damaged or foreign file.
    static std::unique_ptr<SparseIndex> load(const std::string& path);
    static std::unique_ptr<SparseIndex> load(std::istream& in);

    /// A durable index in directory `dir` (durable.h): created if new, or recovered
    /// (the latest snapshot, then the log after it). Every change is logged before it
    /// is applied; a second process opening `dir` is refused while this one holds it.
    static std::unique_ptr<SparseIndex> open_durable(const std::string& dir, Schema schema = Schema::dynamic(),
                                                     const DurableOptions& options = {}, RecoveryStats* stats = nullptr);
    /// Durable indexes: writes a snapshot and drops the log it covers. Returns the
    /// snapshot's LSN. Throws std::logic_error if the index is not durable.
    std::uint64_t checkpoint();
    /// Re-applies logged changes (durable.h recovery). Returns true if the last one,
    /// unfinished when the process stopped, was dropped. Throws FormatError when the log
    /// does not match the index.
    bool replay(std::span<const WalRecord> records);

private:
    struct Posting {
        std::vector<NodeId> slots;  // ascending
        std::vector<float> weights;
    };
    static SparseVector normalized(const SparseVector& v);  // sorted, zeros dropped; throws if invalid
    void insert_unlocked(std::uint64_t id, SparseVector v, const Metadata* metadata, bool keep_metadata);
    bool remove_unlocked(std::uint64_t id);
    void maybe_compact();
    void compact_unlocked();

    MetadataStore metadata_;
    std::unordered_map<std::uint64_t, NodeId> slot_of_;
    std::vector<std::uint64_t> id_of_;      // per slot
    std::vector<std::uint8_t> live_;        // per slot
    std::vector<SparseVector> docs_;        // per slot (sorted); emptied when removed
    std::unordered_map<std::uint32_t, Posting> postings_;
    std::unordered_map<std::uint32_t, std::uint32_t> df_;  // live documents per term
    std::size_t live_count_ = 0;
    std::size_t posting_entries_ = 0;
    std::size_t dead_entries_ = 0;
    mutable IndexLock lock_;
    ChangeLog log_;  // write-ahead log hook (change_log.h)
    std::shared_ptr<detail::DurableState> durable_;  // the directory, if opened with open_durable
    void save_unlocked(std::ostream& out) const;
};

// ---------------------------------------------------------------- BM25

struct Bm25Params {
    double k1 = 1.2;                ///< term-frequency saturation
    double b = 0.75;                ///< length normalization
    double avg_doc_length = 100.0;  ///< in tokens; set it with fit() or from your corpus
    // Text analysis (text_analysis.h), all off by default: tokens as Bm25::tokenize gives
    // them. Fixed when a collection is created: documents and queries must agree.
    bool unicode = false;     ///< Unicode-aware tokens: lowercase beyond ASCII, Unicode punctuation separates, apostrophes kept in words
    bool stop_words = false;  ///< leave out Snowball's English stop words ("the", "and", ...)
    bool stem = false;        ///< reduce words to stems (Snowball English): "running" matches "run"
};

/// BM25 as sparse vectors: documents carry the term-frequency part, queries the
/// IDF part (from the index's document frequencies), so the dot product is the
/// BM25 score. Terms are 32-bit FNV-1a hashes of lowercase tokens (letters and
/// digits; bytes of UTF-8 sequences count as letters); distinct tokens with the
/// same hash merge, which is rare and only blends their counts.
class Bm25 {
public:
    explicit Bm25(Bm25Params params = {});
    static std::vector<std::string> tokenize(std::string_view text);
    /// The terms of `text` as the parameters ask: tokenize() or tokenize_unicode(), then
    /// stop words removed, then stems (of tokens of letters a to z and apostrophes only).
    std::vector<std::string> analyze(std::string_view text) const;
    static std::uint32_t term_id(std::string_view token);
    /// Sets avg_doc_length to the mean token count of `documents` (unchanged if none).
    void fit(std::span<const std::string_view> documents);
    SparseVector encode_document(std::string_view text) const;
    /// IDF weights: ln(1 + (N - df + 0.5) / (df + 0.5)) per query token occurrence,
    /// N and df taken from `index`. Terms no document has are dropped.
    SparseVector encode_query(std::string_view text, const SparseIndex& index) const;
    const Bm25Params& params() const { return params_; }

private:
    Bm25Params params_;
};

// ---------------------------------------------------------------- hybrid search

enum class Fusion {
    Rrf,       ///< sum over lists of weight / (rrf_k + rank), rank from 1
    Weighted,  ///< sum of weight * score, each list's scores scaled to [0, 1] (min-max)
};

struct HybridOptions {
    Fusion fusion = Fusion::Rrf;
    double rrf_k = 60.0;
    double dense_weight = 1.0;
    double sparse_weight = 1.0;
    /// Candidates taken from each side; 0: max(4 * k, 20).
    std::size_t dense_k = 0;
    std::size_t sparse_k = 0;
    /// Beam width for a graph index (HnswIndex); 0: max(dense candidates, 64).
    std::size_t dense_ef = 0;
    /// Applied to both sides (in addition to any filter in the per-side options).
    std::optional<Filter> filter;
    SearchOptions dense_options;
    SearchOptions sparse_options;
    /// Opt-in re-scoring of the fused results by metadata (read from the dense
    /// index): recency decay, confidence multipliers, boosts. See ScoreModifiers.
    std::optional<ScoreModifiers> modifiers;
};

struct HybridResult {
    std::uint64_t id = 0;
    double score = 0;                          ///< fused, higher is better
    std::optional<std::size_t> dense_rank;     ///< 1-based, if found by the dense search
    std::optional<std::size_t> sparse_rank;
    std::optional<float> dense_distance;
    std::optional<float> sparse_score;         ///< the dot product
    std::optional<double> base_score;          ///< the fused score before modifiers (when they apply)
};

/// Fuses two ranked lists (best first, as returned by search()). Ties by ID.
/// Throws std::invalid_argument for negative weights or rrf_k.
std::vector<HybridResult> fuse(std::span<const SearchResult> dense, std::span<const SearchResult> sparse, std::size_t k,
                               const HybridOptions& options = {});

/// Dense search on `dense` (any index with search(query, k, SearchOptions)) and
/// sparse search on `sparse`, fused. The two indexes should hold the same IDs.
template <class DenseIndex>
std::vector<HybridResult> hybrid_search(const DenseIndex& dense, std::span<const float> dense_query, const SparseIndex& sparse,
                                        const SparseVector& sparse_query, std::size_t k, const HybridOptions& options = {}) {
    if (k == 0) return fuse({}, {}, 0, options);  // validates the options; nothing to search for
    // With modifiers, each side retrieves at least the pool they re-score: a boosted
    // document beyond a side's default depth could never be promoted otherwise.
    const std::size_t depth = std::max<std::size_t>(std::max<std::size_t>(4 * k, 20),
                                                    options.modifiers && !options.modifiers->empty() ? options.modifiers->pool_for(k) : 0);
    SearchOptions d = options.dense_options, s = options.sparse_options;
    if (options.filter) {
        d.filter = d.filter ? (*d.filter && *options.filter) : *options.filter;
        s.filter = s.filter ? (*s.filter && *options.filter) : *options.filter;
    }
    const std::size_t dk = options.dense_k ? options.dense_k : depth;
    std::vector<SearchResult> dr;
    if constexpr (requires { dense.search(dense_query, dk, dk, d); })  // a graph index: with a beam width
        dr = dense.search(dense_query, dk, options.dense_ef ? options.dense_ef : std::max<std::size_t>(dk, 64), d);
    else
        dr = dense.search(dense_query, dk, d);
    const auto sr = sparse.search(sparse_query, options.sparse_k ? options.sparse_k : depth, s);
    if (!options.modifiers || options.modifiers->empty()) return fuse(dr, sr, k, options);
    const ScoreModifiers& m = *options.modifiers;  // opt-in: fuse a larger pool, re-score by metadata, keep k
    m.validate();
    std::vector<HybridResult> pool = fuse(dr, sr, m.pool_for(k), options);
    for (HybridResult& r : pool) {
        std::optional<Metadata> md = dense.get_metadata(r.id);
        if (!md) md = sparse.get_metadata(r.id);
        r.base_score = r.score;
        r.score = m.apply(r.score, md ? *md : Metadata{});
    }
    std::stable_sort(pool.begin(), pool.end(), [](const HybridResult& a, const HybridResult& b) { return a.score > b.score; });
    if (pool.size() > k) pool.resize(k);
    return pool;
}

}  // namespace vecdb