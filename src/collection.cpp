/**
 * @file collection.cpp
 * @brief Collections of documents over dense, text and sparse sides; see collection.h.
 */
#include "collection.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "serialize.h"
#include "wal_ops.h"

namespace vecdb {

namespace {

constexpr std::uint32_t kConf = persist::tag("CCFG");
constexpr std::uint32_t kDocs = persist::tag("CDOC");
constexpr std::uint32_t kMeta = persist::tag("CMET");
constexpr std::uint32_t kSide = persist::tag("CSID");

constexpr std::uint8_t kPartVector = 1, kPartText = 2, kPartSparse = 4;

/// Collections searching on this thread. From inside one of their searches (a
/// predicate) a change is refused rather than deadlocking on the lock this thread
/// holds for reading, and reads use that lock instead of taking it again (a
/// recursive shared lock can deadlock once a writer waits).
thread_local std::vector<const void*> searching;

bool inside(const void* c) { return std::find(searching.begin(), searching.end(), c) != searching.end(); }

struct Searching {
    explicit Searching(const void* c) { searching.push_back(c); }
    ~Searching() { searching.pop_back(); }
    Searching(const Searching&) = delete;
    Searching& operator=(const Searching&) = delete;
};

/// A read lock, unless this thread already holds it in a search of this collection.
std::shared_lock<std::shared_mutex> read_lock(std::shared_mutex& m, const void* c) {
    std::shared_lock<std::shared_mutex> l(m, std::defer_lock);
    if (!inside(c)) l.lock();
    return l;
}

/// A write lock; refused from inside a search of this collection.
std::unique_lock<std::shared_mutex> write_lock(std::shared_mutex& m, const void* c) {
    if (inside(c)) throw std::logic_error("Collection: changes from inside its own search (a predicate) are refused");
    return std::unique_lock<std::shared_mutex>(m);
}

template <class Index, class Old>
void restore(Index& index, std::uint64_t id, const std::optional<Old>& old) {
    if (old) (void)index.upsert(id, *old);
    else (void)index.remove(id);
}

std::string to_bytes(const auto& index) {
    std::ostringstream out;
    index.save(out);
    return std::move(out).str();
}

}  // namespace

struct Collection::Prepared {
    std::uint8_t parts = 0;
    std::vector<float> vector;
    std::string text;
    SparseVector text_terms;
    SparseVector sparse;
    Metadata metadata;
};

// ---------------------------------------------------------------- presets

CollectionOptions CollectionOptions::preset(std::size_t dim, Preset preset, std::size_t expected_size) {
    CollectionOptions o;
    o.dim = dim;
    o.metric = Metric::Cosine;
    o.maintenance.mode = Maintenance::Background;
    const bool large = expected_size >= 1000000;  // more links keep recall up in big graphs
    switch (preset) {
        case Preset::Balanced:
            o.hnsw.M = large ? 24 : 16;
            o.hnsw.ef_construction = large ? 300 : 200;
            o.ef = 128;  // the library default (64) measured 0.92 recall@10 on clustered data; 128 reaches 0.95+
            break;
        case Preset::HighRecall:
            o.hnsw.M = large ? 48 : 32;
            o.hnsw.ef_construction = large ? 500 : 400;
            o.ef = 256;
            break;
        case Preset::LowMemory:
            // SQ8 codes only: no full vectors kept, one byte per dimension. (PQ codes
            // alone recall too little without full vectors to re-rank against.)
            o.hnsw.M = 12;
            o.hnsw.ef_construction = 150;
            o.hnsw.quantization.codec = CodecKind::SQ8;
            o.hnsw.quantization.keep_full_vectors = false;
            o.hnsw.quantization.rerank = false;
            o.ef = 128;
            break;
        default: throw std::invalid_argument("CollectionOptions::preset: unknown preset");
    }
    return o;
}

// ---------------------------------------------------------------- construction

Collection::Collection(CollectionOptions options) : options_(std::move(options)), bm25_(options_.bm25), metadata_(options_.schema) {
    if (options_.dim == 0 && !options_.text && !options_.sparse)
        throw std::invalid_argument("Collection: no side at all (set dim, text or sparse)");
    if (options_.dim > 0) dense_ = std::make_unique<HnswIndex>(options_.dim, options_.metric, options_.hnsw);
    if (options_.text) text_ = std::make_unique<SparseIndex>();
    if (options_.sparse) sparse_ = std::make_unique<SparseIndex>();
    checkpoint_time_ = std::chrono::steady_clock::now().time_since_epoch().count();
    const MaintenancePolicy policy = options_.maintenance;
    options_.maintenance = MaintenancePolicy{};
    set_maintenance(policy);  // validates it, and starts the background thread if asked
}

Collection::~Collection() { stop_maintenance(); }  // before any member goes

bool Collection::searching_here() const { return inside(this); }

void Collection::check_changeable() const {
    if (failed_) throw std::runtime_error("Collection: a failed change could not be rolled back; changes are refused until it is reopened");  // GCOVR_EXCL_BR_LINE: needs a second failure while restoring (see apply_upsert)
}

Collection::Prepared Collection::prepare(const Document& d) const {
    Prepared p;
    if (!d.vector.empty()) {
        if (!dense_) throw std::invalid_argument("Collection: a vector given, but the collection has no dense side (dim 0)");
        if (d.vector.size() != options_.dim) throw std::invalid_argument("Collection: vector has the wrong dimension");
        for (const float x : d.vector)
            if (!std::isfinite(x)) throw std::invalid_argument("Collection: vector contains NaN or infinity");
        p.parts |= kPartVector;
        p.vector = d.vector;
    }
    if (!d.text.empty()) {
        if (!text_) throw std::invalid_argument("Collection: text given, but the collection has no text side");
        p.parts |= kPartText;
        p.text = d.text;
        p.text_terms = bm25_.encode_document(d.text);
    }
    if (!d.sparse.empty() || !d.sparse.values.empty()) {
        if (!sparse_) throw std::invalid_argument("Collection: a sparse vector given, but the collection has no sparse side");
        if (d.sparse.indices.size() != d.sparse.values.size()) throw std::invalid_argument("Collection: sparse vector indices and values differ in length");
        std::vector<std::uint32_t> sorted = d.sparse.indices;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) throw std::invalid_argument("Collection: sparse vector repeats a term");
        for (const float w : d.sparse.values)
            if (!std::isfinite(w)) throw std::invalid_argument("Collection: sparse weight is NaN or infinite");
        p.parts |= kPartSparse;
        p.sparse = d.sparse;
    }
    p.metadata = d.metadata;  // checked under the lock: the field table changes with dynamic schemas
    return p;
}

// ---------------------------------------------------------------- changes

bool Collection::apply_upsert(std::uint64_t id, const Prepared& p) {
    const auto it = slot_of_.find(id);
    const bool existed = it != slot_of_.end();
    // The old parts, to restore the sides if a later step fails.
    std::optional<std::vector<float>> old_vector;
    std::optional<SparseVector> old_text, old_sparse;
    Metadata old_metadata;
    if (existed) {
        if (dense_) old_vector = dense_->get_vector(id);
        if (text_) old_text = text_->get_vector(id);
        if (sparse_) old_sparse = sparse_->get_vector(id);
        old_metadata = metadata_.read(it->second);
    }
    NodeId slot = 0;
    if (existed) {
        slot = it->second;
    } else {
        id_of_.reserve(id_of_.size() + 1);
        live_.reserve(live_.size() + 1);
        texts_.reserve(texts_.size() + 1);
        slot = free_.empty() ? static_cast<NodeId>(id_of_.size()) : free_.back();
        slot_of_.emplace(id, slot);  // may throw: nothing changed yet
    }
    std::optional<MetadataStore::WriteUndo> undo;
    std::size_t sides_done = 0;
    try {
        std::string text = options_.store_text ? p.text : std::string();
        undo = metadata_.write(slot, p.metadata);
        if (dense_) {
            if (p.parts & kPartVector) (void)dense_->upsert(id, p.vector);
            else if (old_vector) (void)dense_->remove(id);
        }
        ++sides_done;
        if (text_) {
            if (p.parts & kPartText) (void)text_->upsert(id, p.text_terms);
            else if (old_text) (void)text_->remove(id);
        }
        ++sides_done;
        if (sparse_) {
            if (p.parts & kPartSparse) (void)sparse_->upsert(id, p.sparse);
            else if (old_sparse) (void)sparse_->remove(id);
        }
        // Nothing below throws (reserved above, or swaps).
        if (!existed) {
            if (slot == id_of_.size()) {
                id_of_.push_back(id);
                live_.push_back(1);
                texts_.emplace_back();
            } else {
                free_.pop_back();
                id_of_[slot] = id;
                live_[slot] = 1;
            }
        }
        texts_[slot].swap(text);
        if (old_vector) {  // churn: a dense vector replaced or dropped (additions do not damage the graph)
            ++changed_since_upkeep_;
            ++changed_since_rebuild_;
        }
    } catch (...) {
        try {  // restore the sides already changed, newest first
            if (sides_done >= 2 && text_) restore(*text_, id, old_text);
            if (sides_done >= 1 && dense_) restore(*dense_, id, old_vector);
            if (undo) {
                metadata_.undo_write(slot, *undo);
                if (existed) (void)metadata_.write(slot, old_metadata);
            }
        } catch (...) {
            failed_ = true;  // GCOVR_EXCL_LINE: needs a second failure while restoring; the sides may disagree, so changes are refused until reopened
        }
        if (!existed) slot_of_.erase(id);
        throw;
    }
    return !existed;
}

bool Collection::apply_remove(std::uint64_t id) {
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return false;  // GCOVR_EXCL_BR_LINE: callers pass known IDs only
    const NodeId slot = it->second;
    free_.reserve(free_.size() + 1);
    // Removals never fail on these indexes' side (they free memory).
    if (dense_ && dense_->remove(id)) {
        ++changed_since_upkeep_;
        ++changed_since_rebuild_;
    }
    if (text_) (void)text_->remove(id);
    if (sparse_) (void)sparse_->remove(id);
    metadata_.clear(slot);
    std::string().swap(texts_[slot]);
    live_[slot] = 0;
    free_.push_back(slot);
    slot_of_.erase(it);
    return true;
}

void Collection::add(std::uint64_t id, const Document& document) {
    Prepared p = prepare(document);
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    if (slot_of_.count(id)) throw std::invalid_argument("Collection: duplicate id " + std::to_string(id));
    metadata_.validate(p.metadata);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocument, [&] {
        return walop::encode_document(id, p.parts, p.vector, p.text, p.sparse, p.metadata);
    }));
    (void)apply_upsert(id, p);
}

bool Collection::upsert(std::uint64_t id, const Document& document) {
    Prepared p = prepare(document);
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    metadata_.validate(p.metadata);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocument, [&] {
        return walop::encode_document(id, p.parts, p.vector, p.text, p.sparse, p.metadata);
    }));
    return apply_upsert(id, p);
}

bool Collection::remove(std::uint64_t id) {
    const std::uint64_t one[1] = {id};
    return remove_batch(one) == 1;
}

std::size_t Collection::remove_batch(std::span<const std::uint64_t> ids) {
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    std::vector<std::uint64_t> known;
    for (const std::uint64_t id : ids)
        if (slot_of_.count(id)) known.push_back(id);
    if (known.empty()) return 0;
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocumentRemove, [&] { return walop::encode_ids(known); }));
    std::size_t n = 0;
    for (const std::uint64_t id : known) n += apply_remove(id);
    return n;
}

std::vector<NodeId> Collection::matching(const Filter& filter) const {
    const CompiledFilter compiled(filter, metadata_);
    std::vector<NodeId> out;
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n] && compiled.matches(n)) out.push_back(n);
    return out;
}

std::size_t Collection::remove_where(const Filter& filter) {
    std::vector<std::uint64_t> ids;
    {
        const auto lock = read_lock(mutex_, this);
        for (const NodeId n : matching(filter)) ids.push_back(id_of_[n]);
    }
    // Resolved and removed under one exclusive lock: a document added meanwhile
    // that matches stays (as if it came after), one removed meanwhile is skipped.
    return remove_batch(ids);
}

bool Collection::set_metadata(std::uint64_t id, const Metadata& metadata) {
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return false;
    metadata_.validate(metadata);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocumentMetadata, [&] {
        return walop::encode_set_metadata(std::span<const std::uint64_t>(&id, 1), metadata);
    }));
    metadata_.update(it->second, metadata);
    return true;
}

std::size_t Collection::set_metadata_where(const Filter& filter, const Metadata& changes) {
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    metadata_.validate(changes);
    const std::vector<NodeId> slots = matching(filter);
    if (slots.empty()) return 0;
    std::vector<std::uint64_t> ids;
    for (const NodeId n : slots) ids.push_back(id_of_[n]);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocumentMetadata, [&] { return walop::encode_set_metadata(ids, changes); }));
    MetadataStore staged = metadata_;  // all or nothing
    for (const NodeId n : slots) staged.update(n, changes);
    metadata_ = std::move(staged);
    return slots.size();
}

// ---------------------------------------------------------------- reading

bool Collection::contains(std::uint64_t id) const {
    const auto lock = read_lock(mutex_, this);
    return slot_of_.count(id) != 0;
}

std::size_t Collection::size() const {
    const auto lock = read_lock(mutex_, this);
    return slot_of_.size();
}

std::size_t Collection::count_where(const Filter& filter) const {
    const auto lock = read_lock(mutex_, this);
    return matching(filter).size();
}

std::optional<DocumentRecord> Collection::get(std::uint64_t id) const {
    const auto lock = read_lock(mutex_, this);
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return std::nullopt;
    DocumentRecord r;
    r.id = id;
    if (dense_) r.vector = dense_->get_vector(id);
    if (sparse_) r.sparse = sparse_->get_vector(id);
    r.text = texts_[it->second];
    r.metadata = metadata_.read(it->second);
    return r;
}

std::vector<CollectionHit> Collection::search(const CollectionQuery& q, std::size_t k, const CollectionSearchOptions& o) const {
    const bool use_vector = !q.vector.empty(), use_text = !q.text.empty(), use_sparse = !q.sparse.empty() || !q.sparse.values.empty();
    if (!use_vector && !use_text && !use_sparse) throw std::invalid_argument("Collection::search: the query has no vector, text or sparse vector");
    if (use_vector && !dense_) throw std::invalid_argument("Collection::search: a vector query, but the collection has no dense side");
    if (use_text && !text_) throw std::invalid_argument("Collection::search: a text query, but the collection has no text side");
    if (use_sparse && !sparse_) throw std::invalid_argument("Collection::search: a sparse query, but the collection has no sparse side");
    HybridOptions check;
    check.fusion = o.fusion;
    check.rrf_k = o.rrf_k;
    check.dense_weight = o.vector_weight;
    check.sparse_weight = std::max(o.text_weight, o.sparse_weight);
    (void)fuse({}, {}, 0, check);  // validates rrf_k and weights
    if (!(o.text_weight >= 0) || !(o.sparse_weight >= 0) || !std::isfinite(o.text_weight) || !std::isfinite(o.sparse_weight))
        throw std::invalid_argument("CollectionSearchOptions: weights must be finite and non-negative");
    const auto lock = read_lock(mutex_, this);
    const Searching guard(this);
    if (o.stopped_early) *o.stopped_early = false;
    if (k == 0 || slot_of_.empty()) return {};
    // The filter runs on the collection's metadata, for every side, as a predicate.
    std::optional<CompiledFilter> compiled;
    if (o.filter) compiled.emplace(*o.filter, metadata_);
    SearchOptions side;
    if (compiled || o.predicate) {
        side.predicate = [&](std::uint64_t id) {
            if (compiled) {
                const auto it = slot_of_.find(id);
                if (it == slot_of_.end() || !compiled->matches(it->second)) return false;  // GCOVR_EXCL_BR_LINE: the sides hold collection documents only
            }
            return !o.predicate || o.predicate(id);
        };
    }
    side.deadline = o.deadline;
    side.cancel = o.cancel;
    const std::size_t depth = o.candidates ? o.candidates : std::max<std::size_t>(4 * k, 20);
    struct List {
        std::vector<SearchResult> results;
        double weight;
        int which;  // 0 vector, 1 text, 2 sparse
    };
    std::vector<List> lists;
    const std::size_t ef = o.ef ? o.ef : options_.ef ? std::max(options_.ef, depth) : std::max<std::size_t>(depth, 64);
    bool stopped = false;
    auto run = [&](auto&& search_side) {  // each side reports whether a deadline or cancel stopped it
        SearchStats st;
        SearchOptions with_stats = side;
        with_stats.stats = &st;
        auto results = search_side(with_stats);
        stopped = stopped || st.stopped_early;
        return results;
    };
    if (use_vector) lists.push_back({run([&](const SearchOptions& so) { return dense_->search(q.vector, depth, ef, so); }), o.vector_weight, 0});
    if (use_text) lists.push_back({run([&](const SearchOptions& so) { return text_->search(bm25_.encode_query(q.text, *text_), depth, so); }), o.text_weight, 1});
    if (use_sparse) lists.push_back({run([&](const SearchOptions& so) { return sparse_->search(q.sparse, depth, so); }), o.sparse_weight, 2});
    if (o.stopped_early) *o.stopped_early = stopped;
    // Fusion over any number of lists, as fuse() does for two.
    std::unordered_map<std::uint64_t, CollectionHit> by_id;
    for (const List& l : lists) {
        for (std::size_t i = 0; i < l.results.size(); ++i) {
            CollectionHit& h = by_id[l.results[i].id];
            h.id = l.results[i].id;
            double part = 0;
            if (o.fusion == Fusion::Rrf) part = l.weight / (o.rrf_k + double(i + 1));
            else {
                const double hi = -double(l.results.front().distance), lo = -double(l.results.back().distance);
                part = l.weight * (hi == lo ? 1.0 : (-double(l.results[i].distance) - lo) / (hi - lo));
            }
            h.score += part;
            if (l.which == 0) {
                h.vector_rank = i + 1;
                h.vector_distance = l.results[i].distance;
            } else if (l.which == 1) {
                h.text_rank = i + 1;
                h.text_score = -l.results[i].distance;
            } else {
                h.sparse_rank = i + 1;
                h.sparse_score = -l.results[i].distance;
            }
        }
    }
    std::vector<CollectionHit> out;
    out.reserve(by_id.size());
    for (auto& [id, h] : by_id) out.push_back(std::move(h));
    const std::size_t n = std::min(k, out.size());
    std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), out.end(),
                      [](const CollectionHit& a, const CollectionHit& b) { return a.score > b.score || (a.score == b.score && a.id < b.id); });
    out.resize(n);
    for (CollectionHit& h : out) {
        const NodeId slot = slot_of_.at(h.id);
        if (o.with_metadata) h.metadata = metadata_.read(slot);
        if (o.with_text) h.text = texts_[slot];
    }
    return out;
}

// ---------------------------------------------------------------- replay

bool Collection::replay(std::span<const WalRecord> records) {
    {
        const auto lock = read_lock(mutex_, this);
        if (log_.attached()) throw std::logic_error("replay: the collection's log is attached (replayed changes must not be logged again)");
    }
    std::unordered_set<std::uint64_t> aborted;
    for (const WalRecord& r : records)
        if (r.type == walop::kAbort) aborted.insert(walop::decode(r.type, r.payload).lsn);
    std::uint64_t final_lsn = 0;
    for (const WalRecord& r : records)
        if (r.type != walop::kAbort && !aborted.count(r.lsn)) final_lsn = r.lsn;
    for (const WalRecord& r : records) {
        if (r.type == walop::kAbort || aborted.count(r.lsn)) continue;
        const walop::Op op = walop::decode(r.type, r.payload);
        try {
            switch (op.type) {
                case walop::kDocument: {
                    Document d;
                    d.vector = op.vectors;
                    d.text = op.text;
                    d.sparse = op.sparse;
                    d.metadata = op.metadata.at(0);
                    (void)upsert(op.ids.at(0), d);
                    break;
                }
                case walop::kDocumentRemove: (void)remove_batch(op.ids); break;
                case walop::kDocumentMetadata:
                    for (const std::uint64_t id : op.ids) (void)set_metadata(id, op.metadata.at(0));
                    break;
                default: throw FormatError("log record of a type a collection cannot replay");
            }
        } catch (const FormatError&) {  // GCOVR_EXCL_START: no collection change throws FormatError
            throw;
        }  // GCOVR_EXCL_STOP
        catch (const std::exception& e) {
            if (r.lsn == final_lsn) return true;  // in progress when the process stopped
            throw FormatError("replaying log record " + std::to_string(r.lsn) + " failed: " + e.what() +
                              " (the log does not match the collection it is replayed onto)");
        }
    }
    return false;
}

// ---------------------------------------------------------------- persistence

void Collection::save(std::ostream& out) const {
    const auto lock = read_lock(mutex_, this);
    save_unlocked(out);
}

void Collection::save_unlocked(std::ostream& out) const {
    persist::StreamErrorsThrow errors(out);
    persist::write_header(out, persist::IndexKind::Collection);
    persist::write_section(out, kConf, [&](persist::Writer& w) {
        w.put<std::uint64_t>(options_.dim);
        w.put<std::uint32_t>(static_cast<std::uint32_t>(options_.metric));
        w.put<std::uint8_t>(options_.text ? 1 : 0);
        w.put<std::uint8_t>(options_.sparse ? 1 : 0);
        w.put<std::uint8_t>(options_.store_text ? 1 : 0);
        w.put<double>(options_.bm25.k1);
        w.put<double>(options_.bm25.b);
        w.put<double>(options_.bm25.avg_doc_length);
    });
    persist::write_section(out, kDocs, [&](persist::Writer& w) {
        w.put<std::uint64_t>(id_of_.size());
        for (NodeId n = 0; n < id_of_.size(); ++n) {
            w.put<std::uint8_t>(live_[n]);
            w.put<std::uint64_t>(id_of_[n]);
            w.put_string(texts_[n]);
        }
        w.put<std::uint64_t>(free_.size());  // in order: recovery reuses slots exactly as before
        for (const NodeId n : free_) w.put<std::uint32_t>(n);
    });
    persist::write_section(out, kMeta, [&](persist::Writer& w) { metadata_.save(w); });
    const std::string dense = dense_ ? to_bytes(*dense_) : std::string(), text = text_ ? to_bytes(*text_) : std::string(),
                      sparse = sparse_ ? to_bytes(*sparse_) : std::string();
    persist::write_section(out, kSide, [&](persist::Writer& w) {
        w.put_string(dense);
        w.put_string(text);
        w.put_string(sparse);
    });
}

void Collection::save(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create " + tmp);
        try {
            save(out);
        } catch (...) {  // GCOVR_EXCL_START: needs a stream failing mid-write
            out.close();
            std::remove(tmp.c_str());
            throw;
        }  // GCOVR_EXCL_STOP
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // GCOVR_EXCL_START: a rename onto an existing file fails on some systems
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path);
    }  // GCOVR_EXCL_STOP
}

std::unique_ptr<Collection> Collection::load(std::istream& in) {
    persist::Reader r(in);
    r.read_header(persist::IndexKind::Collection);
    CollectionOptions o;
    r.begin_section(kConf, "configuration");
    const std::uint64_t dim = r.get<std::uint64_t>();
    // Checked before the dense side allocates from it (found by fuzzing: a claimed
    // dimension near 2**63 made the constructor allocate exabytes).
    if (dim > persist::kMaxDim) throw FormatError("collection: invalid dimension");
    o.dim = static_cast<std::size_t>(dim);
    const std::uint32_t metric = r.get<std::uint32_t>();
    if (metric > 2) throw FormatError("collection: unknown metric");
    o.metric = static_cast<Metric>(metric);
    const std::uint8_t text = r.get<std::uint8_t>(), sparse = r.get<std::uint8_t>(), store = r.get<std::uint8_t>();
    if (text > 1 || sparse > 1 || store > 1) throw FormatError("collection: invalid flags");
    o.text = text == 1;
    o.sparse = sparse == 1;
    o.store_text = store == 1;
    o.bm25.k1 = r.get<double>();
    o.bm25.b = r.get<double>();
    o.bm25.avg_doc_length = r.get<double>();
    r.end_section();
    std::unique_ptr<Collection> c;
    try {
        c = std::make_unique<Collection>(o);
    } catch (const std::invalid_argument& e) {
        throw FormatError(std::string("collection: invalid configuration: ") + e.what());
    }
    r.begin_section(kDocs, "documents");
    const std::uint64_t slots = r.get_count(13, "documents");
    c->id_of_.reserve(slots);
    for (std::uint64_t n = 0; n < slots; ++n) {
        const std::uint8_t live = r.get<std::uint8_t>();
        if (live > 1) throw FormatError("collection: invalid live flag");
        const std::uint64_t id = r.get<std::uint64_t>();
        std::string t = r.get_string();
        if (live && !c->slot_of_.emplace(id, static_cast<NodeId>(n)).second) throw FormatError("collection: duplicate id");
        c->id_of_.push_back(id);
        c->live_.push_back(live);
        c->texts_.push_back(std::move(t));
    }
    const std::uint64_t free = r.get_count(4, "free slots");
    std::vector<std::uint8_t> seen(c->id_of_.size(), 0);
    for (std::uint64_t i = 0; i < free; ++i) {
        const std::uint32_t n = r.get<std::uint32_t>();
        if (n >= c->id_of_.size() || c->live_[n] || seen[n]) throw FormatError("collection: invalid free slot");
        seen[n] = 1;
        c->free_.push_back(n);
    }
    if (c->free_.size() + c->slot_of_.size() != c->id_of_.size()) throw FormatError("collection: free slots do not match");
    r.end_section();
    r.begin_section(kMeta, "metadata");
    c->metadata_ = MetadataStore::load(r, static_cast<std::size_t>(slots), [&](NodeId n) { return c->live_[n] != 0; });
    r.end_section();
    r.begin_section(kSide, "sides");
    const std::string dense = r.get_string(), text_bytes = r.get_string(), sparse_bytes = r.get_string();
    r.end_section();
    r.expect_end_of_file();
    auto side = [](const std::string& bytes, bool present, auto load) {
        if (!present) {
            if (!bytes.empty()) throw FormatError("collection: data for a side it does not have");
            return decltype(load(std::declval<std::istream&>()))();
        }
        std::istringstream s(bytes);
        return load(s);
    };
    c->dense_ = side(dense, o.dim > 0, [](std::istream& s) { return HnswIndex::load(s); });
    c->text_ = side(text_bytes, o.text, [](std::istream& s) { return SparseIndex::load(s); });
    c->sparse_ = side(sparse_bytes, o.sparse, [](std::istream& s) { return SparseIndex::load(s); });
    if (c->dense_ && (c->dense_->dim() != o.dim || c->dense_->metric() != o.metric)) throw FormatError("collection: dense side does not match");
    // Every document of a side must be a document of the collection.
    auto check = [&](auto& index_records_ids) {
        for (const std::uint64_t id : index_records_ids)
            if (!c->slot_of_.count(id)) throw FormatError("collection: a side holds an unknown document");
    };
    if (c->text_) {
        std::vector<std::uint64_t> ids;
        for (const auto& rec : c->text_->records()) ids.push_back(rec.id);
        check(ids);
    }
    if (c->sparse_) {
        std::vector<std::uint64_t> ids;
        for (const auto& rec : c->sparse_->records()) ids.push_back(rec.id);
        check(ids);
    }
    if (c->dense_) {
        std::size_t found = 0;
        for (const auto& [id, slot] : c->slot_of_) found += c->dense_->contains(id);
        if (found != c->dense_->size()) throw FormatError("collection: the dense side holds an unknown document");
    }
    return c;
}

std::unique_ptr<Collection> Collection::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return load(in);
}

// ---------------------------------------------------------------- maintenance

namespace {

std::string fmt(const char* f, double a, double b) {
    char buf[160];
    std::snprintf(buf, sizeof buf, f, a, b);
    return buf;
}

}  // namespace

void Collection::set_maintenance(const MaintenancePolicy& policy) {
    const auto non_negative = [](double x) { return std::isfinite(x) && x >= 0; };
    if (policy.interval.count() < 1 || policy.budget.count() < 0 || policy.checkpoint_age.count() < 0 || !non_negative(policy.health_check_churn) ||
        !non_negative(policy.upkeep_churn) || !non_negative(policy.rebuild_check_churn) || !non_negative(policy.rebuild_drift))
        throw std::invalid_argument("MaintenancePolicy: interval of at least 1 ms, and non-negative budget, age and thresholds required");
    if (searching_here()) throw std::logic_error("set_maintenance: refused from inside a search of this collection");
    stop_maintenance();
    {
        const std::lock_guard<std::mutex> one(maintenance_mutex_);
        options_.maintenance = policy;
    }
    if (policy.mode != Maintenance::Background) return;
    stopping_ = false;
    maintainer_ = std::thread([this, interval = policy.interval, budget = policy.budget] {
        std::unique_lock<std::mutex> l(wake_mutex_);
        while (!stopping_) {
            wake_.wait_for(l, interval, [this] { return stopping_; });
            if (stopping_) break;
            l.unlock();
            try {
                (void)maintain(budget);
            } catch (const std::exception& e) {  // recorded; the next round tries again
                MaintenanceReport r;
                r.when = std::chrono::system_clock::now();
                r.error = e.what();
                const std::lock_guard<std::mutex> rl(report_mutex_);
                last_report_ = std::move(r);
                ++rounds_;
            }
            l.lock();
        }
    });
}

void Collection::stop_maintenance() {
    {
        const std::lock_guard<std::mutex> l(wake_mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (maintainer_.joinable()) maintainer_.join();
}

std::optional<MaintenanceReport> Collection::last_maintenance() const {
    const std::lock_guard<std::mutex> l(report_mutex_);
    return last_report_;
}

MaintenanceStatus Collection::maintenance_status() const {
    MaintenanceStatus st;
    const auto lock = read_lock(mutex_, this);
    if (durable_) {
        st.log_bytes = log_.writer().size() - wal::kHeaderSize;
        st.log_records = log_.writer().next_lsn() - checkpoint_lsn_.load();
    }
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    st.since_checkpoint = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::nanoseconds(now - checkpoint_time_.load()));
    const double live = double(std::max<std::size_t>(dense_ ? dense_->size() : 0, 1));
    st.churn_since_upkeep = double(changed_since_upkeep_.load()) / live;
    st.churn_since_rebuild = double(changed_since_rebuild_.load()) / live;
    {
        const std::lock_guard<std::mutex> l(report_mutex_);
        st.rounds = rounds_;
    }
    return st;
}

MaintenanceReport Collection::maintain(std::chrono::milliseconds budget) {
    if (searching_here()) throw std::logic_error("maintain: refused from inside a search of this collection");
    const std::lock_guard<std::mutex> one(maintenance_mutex_);
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    std::optional<clock::time_point> deadline;
    if (budget.count() > 0) deadline = start + budget;
    const auto out_of_time = [&] { return deadline && clock::now() >= *deadline; };
    const MaintenancePolicy p = options_.maintenance;
    MaintenanceReport r;
    r.when = std::chrono::system_clock::now();
    if (dense_) {
        const double live = double(std::max<std::size_t>(dense_->size(), 1));
        // Upkeep: past one threshold always; past a lower one if a health check finds damage.
        const std::uint64_t seen = changed_since_upkeep_.load();
        const double churn = double(seen) / live;
        bool upkeep = false;
        if (seen > 0 && churn >= p.upkeep_churn) {
            upkeep = true;
            r.notes.push_back(fmt("upkeep due: churn %.3f reached %.3f", churn, p.upkeep_churn));
        } else if (seen > 0 && churn >= p.health_check_churn) {
            if (out_of_time()) {  // GCOVR_EXCL_BR_LINE: nothing runs before it; only with a budget of microseconds
                r.budget_exhausted = true;  // GCOVR_EXCL_LINE: as above
                r.notes.push_back("health check due, but the budget is spent");  // GCOVR_EXCL_LINE: as above
            } else {
                HealthOptions ho;
                ho.recall_sample = 0;
                ho.codec_sample = 0;
                r.health = dense_->health(ho);
                const std::size_t damage = r.health->unreachable + r.health->links_to_removed + r.health->stale_links;
                if (damage > 0) {
                    upkeep = true;
                    r.notes.push_back("upkeep due: health check found " + std::to_string(r.health->unreachable) + " unreachable vectors, " +
                                      std::to_string(r.health->links_to_removed) + " links to removed vectors, " +
                                      std::to_string(r.health->stale_links) + " stale links");
                } else {
                    changed_since_upkeep_ -= seen;
                    r.notes.push_back(fmt("health check at churn %.3f: nothing to repair", churn, 0));
                }
            }
        } else if (seen > 0) {
            r.notes.push_back(fmt("no upkeep: churn %.3f below %.3f", churn, p.health_check_churn));
        }
        if (upkeep) {
            if (out_of_time()) {
                r.budget_exhausted = true;
                r.notes.push_back("upkeep due, but the budget is spent");
            } else {
                UpkeepOptions uo;
                uo.max_nodes = p.upkeep_nodes;
                OperationOptions oo;
                oo.deadline = deadline;
                r.upkeep = dense_->upkeep(uo, oo);
                r.upkept = true;
                if (r.upkeep.complete) {
                    changed_since_upkeep_ -= seen;
                    r.notes.push_back("upkeep: " + std::to_string(r.upkeep.nodes_repaired) + " vectors repaired, " +
                                      std::to_string(r.upkeep.reconnected) + " reconnected");
                } else {
                    r.budget_exhausted = true;
                    r.notes.push_back("upkeep stopped by the budget or node limit; the next round continues");
                }
            }
        }
        // Rebuild: only with a codec, once enough changed for it to drift.
        if (options_.hnsw.quantization.codec != CodecKind::None) {
            const std::uint64_t seen_r = changed_since_rebuild_.load();
            const double churn_r = double(seen_r) / live;
            if (seen_r > 0 && churn_r >= p.rebuild_check_churn) {
                if (out_of_time()) {  // GCOVR_EXCL_BR_LINE: timing (only when upkeep just used the whole budget)
                    r.budget_exhausted = true;  // GCOVR_EXCL_LINE: as above
                    r.notes.push_back("codec drift check due, but the budget is spent");  // GCOVR_EXCL_LINE: as above
                } else {
                    HealthOptions ho;
                    ho.recall_sample = 0;
                    ho.codec_sample = 500;
                    ho.retrain_sample = 500;
                    const HealthReport h = dense_->health(ho);
                    r.health = h;
                    const bool drifted = h.codec_error && h.codec_error_retrained && *h.codec_error_retrained > 0 &&
                                         *h.codec_error > p.rebuild_drift * *h.codec_error_retrained;
                    if (drifted) {
                        r.notes.push_back(fmt("rebuild due: codec error %.3g is over the drift limit (retrained: %.3g)", *h.codec_error,
                                              *h.codec_error_retrained));
                        r.rebuild = dense_->rebuild();
                        r.rebuilt = true;
                    } else {
                        r.notes.push_back("codec drift check: within the limit");
                    }
                    changed_since_rebuild_ -= seen_r;
                }
            }
        }
    }
    if (durable_) {
        std::uint64_t bytes = 0, records = 0;
        {
            const auto lock = read_lock(mutex_, this);
            bytes = log_.writer().size() - wal::kHeaderSize;
            records = log_.writer().next_lsn() - checkpoint_lsn_.load();
        }
        const auto age = std::chrono::nanoseconds(clock::now().time_since_epoch().count() - checkpoint_time_.load());
        std::string why;
        if (bytes >= p.checkpoint_bytes && bytes > 0) why = "the log reached " + std::to_string(bytes) + " bytes";
        else if (records >= p.checkpoint_records && records > 0) why = "the log reached " + std::to_string(records) + " records";
        else if (records > 0 && age >= p.checkpoint_age) why = "records have waited " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(age).count()) + " s";
        if (!why.empty()) {
            r.checkpoint_lsn = checkpoint();
            r.checkpointed = true;
            r.notes.push_back("checkpoint: " + why);
        }
    }
    r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
    const std::lock_guard<std::mutex> l(report_mutex_);
    last_report_ = r;
    ++rounds_;
    return r;
}

}  // namespace vecdb