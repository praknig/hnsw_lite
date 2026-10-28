/**
 * @file collection.cpp
 * @brief Collections of documents over dense, text and sparse sides; see collection.h.
 */
#include "common.h"
#include "scoring.h"
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

namespace {

constexpr const char* kExpiresField = "_expires_at";
constexpr const char* kKeyField = "_key";

/// A strict schema gains the reserved fields of the features turned on.
CollectionOptions with_reserved_fields(CollectionOptions o) {
    if (o.schema.mode() != SchemaMode::Strict) return o;
    std::vector<FieldSpec> fields = o.schema.fields();
    auto add = [&](const char* name, FieldType type) {
        for (const FieldSpec& f : fields)
            if (f.name == name) return;
        fields.push_back({name, type});
    };
    if (o.ttl.enabled) add(kExpiresField, FieldType::Int);
    if (o.string_ids) add(kKeyField, FieldType::Keyword);
    o.schema = Schema::strict(std::move(fields));
    return o;
}

void check_unreserved(const Metadata& md) {
    for (const auto& [name, value] : md.fields())
        if (name == kExpiresField || name == kKeyField)
            throw std::invalid_argument("Collection: the metadata field " + name + " is reserved (expiry and string ids)");
}

}  // namespace

Collection::Collection(CollectionOptions options)
    : options_(with_reserved_fields(std::move(options))), bm25_(options_.bm25), metadata_(options_.schema) {
    if (options_.ttl.default_ms < 0) throw std::invalid_argument("TtlOptions::default_ms must not be negative");
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

CollectionOptions Collection::options() const {
    const auto lock = read_lock(mutex_, this);
    CollectionOptions o = options_;
    o.schema = metadata_.current_schema();  // as stored now: a dynamic schema gains fields with every new one
    return o;
}

Schema Collection::schema() const {
    const auto lock = read_lock(mutex_, this);
    return metadata_.current_schema();
}

Collection::Runtime Collection::runtime() const {
    const auto lock = read_lock(mutex_, this);
    return {options_.ttl, options_.string_ids};
}

Collection::Prepared Collection::prepare(const Document& d) const { return prepare_keyed(d, std::nullopt, runtime()); }

Collection::Prepared Collection::prepare_keyed(const Document& d, std::optional<std::string_view> key, const Runtime& rt) const {
    if (!replaying_) check_unreserved(d.metadata);  // a logged document carries its reserved fields
    if (d.expires_at && !rt.ttl.enabled) throw std::invalid_argument("Collection: expires_at needs CollectionOptions::ttl.enabled");
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
    if (rt.ttl.enabled && !replaying_) {  // the expiry time, fixed now, is logged with the document
        if (d.expires_at) p.metadata.set(kExpiresField, *d.expires_at);
        else if (rt.ttl.default_ms > 0) p.metadata.set(kExpiresField, now_ms() + rt.ttl.default_ms);
    }
    if (key) p.metadata.set(kKeyField, std::string(*key));
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
        reserve_one_more(id_of_);
        reserve_one_more(live_);
        reserve_one_more(texts_);
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
    reserve_one_more(free_);
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
    const MetadataStore::BulkRemoval bulk(metadata_, known.size());
    for (const std::uint64_t id : known) n += apply_remove(id);
    return n;
}

std::vector<NodeId> Collection::matching(const Filter& filter) const {
    const CompiledFilter compiled(filter, metadata_);
    std::vector<NodeId> out;
    if (const auto rows = compiled.candidates()) {  // posting lists: only rows that can match
        for (const NodeId n : *rows)
            if (n < id_of_.size() && live_[n] && compiled.matches(n)) out.push_back(n);
        return out;
    }
    std::vector<std::uint8_t> match;  // every row: the filter a column at a time
    compiled.matches_all(static_cast<NodeId>(id_of_.size()), match);
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n] && match[n]) out.push_back(n);
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
    if (!replaying_) check_unreserved(metadata);
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
    if (!replaying_) check_unreserved(changes);
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
    const auto it = slot_of_.find(id);
    return it != slot_of_.end() && !expired(it->second);
}

std::size_t Collection::size() const {
    const auto lock = read_lock(mutex_, this);
    return slot_of_.size();
}

std::vector<std::string> Collection::payload_indexes() const {
    const auto lock = read_lock(mutex_, this);
    return metadata_.indexed_fields();
}

void Collection::create_payload_index(std::string_view field) {
    const auto lock = write_lock(mutex_, this);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kCreatePayloadIndex, [&] { return walop::encode_field(field); }));
    metadata_.index_field(field);
}

std::vector<FacetResult> Collection::facets(const std::vector<FacetRequest>& requests, const std::optional<Filter>& filter) const {
    const auto lock = read_lock(mutex_, this);
    const std::optional<Filter> alive = alive_filter();  // expiry: expired documents are never counted
    if (filter && alive) return count_facets(metadata_, requests, matching(*filter && *alive));
    if (filter || alive) return count_facets(metadata_, requests, matching(filter ? *filter : *alive));  // counts include expired rows: read them
    const std::function<std::vector<NodeId>()> all_live = [&] {
        std::vector<NodeId> slots;
        for (NodeId n = 0; n < id_of_.size(); ++n)
            if (live_[n]) slots.push_back(n);
        return slots;
    };
    std::size_t live = 0;
    for (NodeId n = 0; n < id_of_.size(); ++n) live += live_[n] ? 1 : 0;
    return count_facets(metadata_, requests, live, all_live);
}

std::vector<FacetResult> Collection::facets_for(std::span<const std::uint64_t> ids, const std::vector<FacetRequest>& requests) const {
    const auto lock = read_lock(mutex_, this);
    std::vector<NodeId> slots;
    std::unordered_set<std::uint64_t> seen;
    for (const std::uint64_t id : ids)
        if (seen.insert(id).second)
            if (const auto it = slot_of_.find(id); it != slot_of_.end() && !expired(it->second)) slots.push_back(it->second);
    return count_facets(metadata_, requests, slots);
}

std::size_t Collection::count_where(const Filter& filter) const {
    const auto lock = read_lock(mutex_, this);
    const std::optional<Filter> alive = alive_filter();
    return matching(alive ? filter && *alive : filter).size();
}

std::optional<DocumentRecord> Collection::get(std::uint64_t id) const {
    const auto lock = read_lock(mutex_, this);
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end() || expired(it->second)) return std::nullopt;
    return record_unlocked(it->second, true, true, true);
}

DocumentRecord Collection::record_unlocked(NodeId slot, bool vectors, bool text, bool metadata) const {
    DocumentRecord r;
    r.id = id_of_[slot];
    if (vectors && dense_) r.vector = dense_->get_vector(r.id);
    if (vectors && sparse_) r.sparse = sparse_->get_vector(r.id);
    if (text) r.text = texts_[slot];
    r.metadata = metadata_.read(slot);
    fill_reserved(r.metadata, r.key, &r.expires_at);  // the key and expiry time come back either way
    if (!metadata) r.metadata = Metadata();
    return r;
}

CollectionPage Collection::scan(const CollectionScanOptions& options) const {
    const auto lock = read_lock(mutex_, this);
    if (options.limit == 0) throw std::invalid_argument("scan: limit must be at least 1");
    const std::optional<Filter> alive = alive_filter();  // expiry: expired documents are never listed
    std::optional<Filter> filter = options.filter;
    if (alive) filter = filter ? *filter && *alive : *alive;
    // Every candidate after the cursor; then the `limit` smallest IDs, in order.
    std::vector<std::pair<std::uint64_t, NodeId>> candidates;
    auto consider = [&](NodeId n) {
        if (!options.after || id_of_[n] > *options.after) candidates.emplace_back(id_of_[n], n);
    };
    if (filter) {
        for (const NodeId n : matching(*filter)) consider(n);  // throws for an unknown field under a strict schema
    } else {
        for (NodeId n = 0; n < id_of_.size(); ++n)
            if (live_[n]) consider(n);
    }
    const bool more = candidates.size() > options.limit;
    if (more) {
        std::nth_element(candidates.begin(), candidates.begin() + std::ptrdiff_t(options.limit), candidates.end());
        candidates.resize(options.limit);
    }
    std::sort(candidates.begin(), candidates.end());
    CollectionPage page;
    page.documents.reserve(candidates.size());
    for (const auto& [id, n] : candidates)
        page.documents.push_back(record_unlocked(n, options.with_vectors, options.with_text, options.with_metadata));
    if (more) page.next = page.documents.back().id;
    return page;
}

void Collection::check_query(const CollectionQuery& q, const CollectionSearchOptions& o) const {
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
}

std::vector<CollectionHit> Collection::search(const CollectionQuery& q, std::size_t k, const CollectionSearchOptions& o) const {
    check_query(q, o);
    const auto lock = read_lock(mutex_, this);
    const Searching guard(this);
    return search_unlocked(q, k, o);
}

std::vector<CollectionHit> Collection::search_unlocked(const CollectionQuery& q, std::size_t k, const CollectionSearchOptions& o) const {
    const bool use_vector = !q.vector.empty(), use_text = !q.text.empty(), use_sparse = !q.sparse.empty() || !q.sparse.values.empty();
    if (o.stopped_early) *o.stopped_early = false;
    if (k == 0 || slot_of_.empty()) return {};
    // The filter runs on the collection's metadata, for every side, as a predicate.
    std::optional<CompiledFilter> compiled;
    const std::optional<Filter> alive = alive_filter();  // expiry: expired documents are never found
    if (o.filter && alive) compiled.emplace(*o.filter && *alive, metadata_);
    else if (o.filter) compiled.emplace(*o.filter, metadata_);
    else if (alive) compiled.emplace(*alive, metadata_);
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
    if (o.modifiers && !o.modifiers->empty()) o.modifiers->validate();
    if (o.diversity && !(*o.diversity >= 0.0 && *o.diversity <= 1.0)) throw std::invalid_argument("CollectionSearchOptions::diversity must be in [0, 1]");
    if (o.diversity && !dense_) throw std::invalid_argument("CollectionSearchOptions::diversity needs a dense side (the documents' vectors)");
    const std::size_t depth = std::max<std::size_t>(o.candidates ? o.candidates : std::max<std::size_t>(4 * k, 20), o.modifiers && !o.modifiers->empty() ? o.modifiers->pool_for(k) : 0);
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
    // Fusion over any number of lists, as fuse() does for two: candidates as light records
    // (ranked, re-scored and ordered as such); full hits only for those returned.
    struct Fused {
        std::uint64_t id;
        double score = 0;
        double base = 0;                       // before modifiers
        std::uint32_t rank[3] = {0, 0, 0};     // 1-based per side; 0: not in that list
        float side[3] = {0, 0, 0};             // vector distance, text score, sparse score
    };
    std::vector<Fused> fused;
    std::unordered_map<std::uint64_t, std::uint32_t> at;  // ID -> its record
    std::size_t total = 0;
    for (const List& l : lists) total += l.results.size();
    fused.reserve(total);
    at.reserve(total);
    for (const List& l : lists) {
        for (std::size_t i = 0; i < l.results.size(); ++i) {
            const auto [it, fresh] = at.try_emplace(l.results[i].id, static_cast<std::uint32_t>(fused.size()));
            if (fresh) fused.push_back(Fused{l.results[i].id});
            Fused& h = fused[it->second];
            double part = 0;
            if (o.fusion == Fusion::Rrf) part = l.weight / (o.rrf_k + double(i + 1));
            else {
                const double hi = -double(l.results.front().distance), lo = -double(l.results.back().distance);
                part = l.weight * (hi == lo ? 1.0 : (-double(l.results[i].distance) - lo) / (hi - lo));
            }
            h.score += part;
            h.rank[l.which] = static_cast<std::uint32_t>(i + 1);
            h.side[l.which] = l.which == 0 ? l.results[i].distance : -l.results[i].distance;
        }
    }
    const bool modified = o.modifiers && !o.modifiers->empty();
    if (modified) {  // opt-in: every fused candidate re-scored by its metadata
        for (Fused& h : fused) {
            h.base = h.score;
            h.score = o.modifiers->apply(h.score, metadata_.read(slot_of_.at(h.id)));
        }
    }
    const std::size_t n = std::min(k, fused.size());
    const auto better = [](const Fused& a, const Fused& b) { return a.score > b.score || (a.score == b.score && a.id < b.id); };
    if (o.diversity) {  // opt-in: maximal marginal relevance over every fused candidate
        std::sort(fused.begin(), fused.end(), better);
        std::vector<double> relevance(fused.size());
        std::vector<std::optional<std::vector<float>>> vectors(fused.size());
        for (std::size_t i = 0; i < fused.size(); ++i) {
            relevance[i] = fused[i].score;
            vectors[i] = dense_->get_vector(fused[i].id);
        }
        const std::vector<std::size_t> order = mmr_order(relevance, vectors, n, *o.diversity);
        std::vector<Fused> chosen;
        chosen.reserve(order.size());
        for (const std::size_t i : order) chosen.push_back(fused[i]);
        fused.swap(chosen);
    } else {
        std::partial_sort(fused.begin(), fused.begin() + static_cast<std::ptrdiff_t>(n), fused.end(), better);
        fused.resize(n);
    }
    std::vector<CollectionHit> out(fused.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        const Fused& f = fused[i];
        CollectionHit& h = out[i];
        h.id = f.id;
        h.score = f.score;
        if (modified) h.base_score = f.base;
        if (f.rank[0]) {
            h.vector_rank = f.rank[0];
            h.vector_distance = f.side[0];
        }
        if (f.rank[1]) {
            h.text_rank = f.rank[1];
            h.text_score = f.side[1];
        }
        if (f.rank[2]) {
            h.sparse_rank = f.rank[2];
            h.sparse_score = f.side[2];
        }
    }
    for (CollectionHit& h : out) {
        const NodeId slot = slot_of_.at(h.id);
        if (o.with_metadata || options_.string_ids) {
            Metadata md = metadata_.read(slot);
            fill_reserved(md, h.key, nullptr);
            if (o.with_metadata) h.metadata = std::move(md);
        }
        if (o.with_text) h.text = texts_[slot];
    }
    return out;
}

std::vector<CollectionHit> Collection::recommend(const Recommend& r, std::size_t k, const CollectionSearchOptions& o) const {
    if (r.positive.empty()) throw std::invalid_argument("recommend: give at least one positive example");
    if (!dense_) throw std::invalid_argument("recommend: the collection has no vectors");
    std::unordered_set<std::uint64_t> examples(r.positive.begin(), r.positive.end());
    for (const std::uint64_t id : r.negative)  // the set holds the positives so far
        if (examples.count(id))
            throw std::invalid_argument("recommend: document " + std::to_string(id) + " is both a positive and a negative example");
    examples.insert(r.negative.begin(), r.negative.end());
    const auto lock = read_lock(mutex_, this);
    const Searching guard(this);
    // The examples' vectors, read with the search under one lock; averaged in double.
    const std::size_t d = options_.dim;
    auto average = [&](const std::vector<std::uint64_t>& ids) {
        std::vector<double> sum(d, 0.0);
        for (const std::uint64_t id : ids) {
            const auto it = slot_of_.find(id);
            if (it == slot_of_.end() || expired(it->second)) throw std::invalid_argument("recommend: unknown document " + std::to_string(id));
            const DocumentRecord rec = record_unlocked(it->second, true, false, false);
            if (!rec.vector) throw std::invalid_argument("recommend: document " + std::to_string(id) + " has no vector");
            double norm = 0;
            for (const float x : *rec.vector) norm += double(x) * x;
            norm = std::sqrt(norm);
            const double scale = options_.metric == Metric::Cosine && norm > 0 ? 1.0 / norm : 1.0;  // cosine: each example counts alike
            for (std::size_t j = 0; j < d; ++j) sum[j] += double((*rec.vector)[j]) * scale;
        }
        for (double& x : sum) x /= double(ids.size());
        return sum;
    };
    const std::vector<double> pos = average(r.positive);
    const std::vector<double> neg = r.negative.empty() ? std::vector<double>{} : average(r.negative);
    CollectionQuery q;
    q.vector.resize(d);
    for (std::size_t j = 0; j < d; ++j) {
        const double v = neg.empty() ? pos[j] : 2.0 * pos[j] - neg[j];  // away from the negatives
        constexpr double kMax = double(std::numeric_limits<float>::max());
        q.vector[j] = static_cast<float>(std::clamp(v, -kMax, kMax));  // finite: 2 max - (-max) would overflow a float
    }
    check_query(q, o);
    // The examples take places in the ranking: ask for that many more, then drop them.
    const std::size_t wanted = k > std::numeric_limits<std::size_t>::max() - examples.size() ? std::numeric_limits<std::size_t>::max() : k + examples.size();
    std::vector<CollectionHit> hits = search_unlocked(q, wanted, o);
    std::erase_if(hits, [&](const CollectionHit& h) { return examples.count(h.id) != 0; });
    if (hits.size() > k) hits.resize(k);
    return hits;
}

std::vector<CollectionGroup> Collection::search_groups(const CollectionQuery& q, const GroupBy& g, const CollectionSearchOptions& o) const {
    if (g.field.empty()) throw std::invalid_argument("search_groups: the field name is empty");
    check_query(q, o);
    const auto lock = read_lock(mutex_, this);
    const Searching guard(this);
    if (g.groups == 0 || g.per_group == 0 || slot_of_.empty()) return {};
    const auto f = metadata_.find_field(g.field);
    if (!f) {
        if (metadata_.current_schema().mode() == SchemaMode::Strict) throw std::invalid_argument("search_groups: unknown field " + g.field);
        return {};  // a dynamic schema: no document holds it yet
    }
    const FieldType type = metadata_.field_type(*f);
    if (type != FieldType::Keyword && type != FieldType::Int && type != FieldType::Bool)
        throw std::invalid_argument("search_groups: group by a keyword, integer or boolean field (" + g.field + " is not)");
    const std::size_t cap = g.max_candidates ? std::min(g.max_candidates, slot_of_.size()) : slot_of_.size();
    std::size_t fetch = std::min(cap, std::max<std::size_t>(16, saturating_mul(saturating_mul(g.groups, g.per_group), 2)));
    std::vector<CollectionGroup> out;
    while (true) {
        const std::vector<CollectionHit> hits = search_unlocked(q, fetch, o);
        out.clear();
        std::unordered_map<std::uint64_t, std::size_t> where;  // a value's group, by position
        for (const CollectionHit& h : hits) {
            const NodeId slot = slot_of_.at(h.id);
            if (!metadata_.has(*f, slot)) continue;  // without the field: in no group
            const std::uint64_t key = type == FieldType::Int    ? static_cast<std::uint64_t>(metadata_.int_at(*f, slot))
                                      : type == FieldType::Bool ? std::uint64_t(metadata_.bool_at(*f, slot))
                                                                : std::uint64_t(metadata_.keyword_at(*f, slot));
            auto it = where.find(key);
            if (it == where.end()) {
                if (out.size() == g.groups) continue;  // a value beyond the groups wanted
                it = where.emplace(key, out.size()).first;
                out.push_back({type == FieldType::Int    ? Value::of_int(metadata_.int_at(*f, slot))
                               : type == FieldType::Bool ? Value::of_bool(metadata_.bool_at(*f, slot))
                                                         : Value::of_keyword(metadata_.string_at(metadata_.keyword_at(*f, slot))),
                               {}});
            }
            std::vector<CollectionHit>& group = out[it->second].hits;
            if (group.size() < g.per_group) group.push_back(h);
        }
        bool full = out.size() == g.groups;
        for (const CollectionGroup& group : out) full = full && group.hits.size() == g.per_group;
        const bool exhausted = hits.size() < fetch;  // every match was ranked
        const bool stopped = o.stopped_early && *o.stopped_early;
        if (full || exhausted || stopped || fetch >= cap) break;
        fetch = std::min(cap, saturating_mul(fetch, 4));
    }
    return out;
}

// ---------------------------------------------------------------- replay

bool Collection::replay(std::span<const WalRecord> records) {
    {
        const auto lock = read_lock(mutex_, this);
        if (log_.attached()) throw std::logic_error("replay: the collection's log is attached (replayed changes must not be logged again)");
    }
    struct Replaying {  // documents are applied exactly as they were logged
        bool& flag;
        explicit Replaying(bool& f) : flag(f) { flag = true; }
        ~Replaying() { flag = false; }
    } replaying(replaying_);
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
                case walop::kCreatePayloadIndex: metadata_.index_field(op.field); break;
                case walop::kCollectionSettings: {  // applied as logged (the log is not attached while replaying)
                    const TtlOptions ttl{op.ttl_enabled == 1, op.ttl_default_ms};
                    check_runtime(ttl, op.string_ids == 1);
                    options_.ttl = ttl;
                    options_.string_ids = op.string_ids == 1;
                    break;
                }
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
        // The runtime settings, only when they differ from the defaults: a collection
        // that uses neither expiry nor string keys is written exactly as before, so
        // older versions still read it (they refuse a file with the settings, cleanly).
        if (options_.ttl.enabled || options_.ttl.default_ms != 0 || options_.string_ids) {
            w.put<std::uint8_t>(1);  // the settings block's own version
            w.put<std::uint8_t>(options_.ttl.enabled ? 1 : 0);
            w.put<std::int64_t>(options_.ttl.default_ms);
            w.put<std::uint8_t>(options_.string_ids ? 1 : 0);
        }
        // Text analysis likewise: only when used, so other collections are written as before.
        const Bm25Params& t = options_.bm25;
        if (t.unicode || t.stop_words || t.stem) {
            w.put<std::uint8_t>(2);  // the text-analysis block
            w.put<std::uint8_t>(static_cast<std::uint8_t>((t.unicode ? 1 : 0) | (t.stop_words ? 2 : 0) | (t.stem ? 4 : 0)));
        }
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
    std::uint8_t last_block = 0;
    while (r.section_left() > 0) {  // optional blocks, each written only when not the defaults, in order
        const std::uint8_t block = r.get<std::uint8_t>();
        if (block <= last_block || block > 2) throw FormatError("collection: unknown settings block");
        last_block = block;
        if (block == 1) {  // runtime settings
            const std::uint8_t ttl = r.get<std::uint8_t>();
            const std::int64_t default_ms = r.get<std::int64_t>();
            const std::uint8_t keys = r.get<std::uint8_t>();
            if (ttl > 1 || keys > 1 || default_ms < 0) throw FormatError("collection: invalid settings");
            o.ttl = TtlOptions{ttl == 1, default_ms};
            o.string_ids = keys == 1;
        } else {  // text analysis
            const std::uint8_t flags = r.get<std::uint8_t>();
            if (flags == 0 || flags > 7) throw FormatError("collection: invalid text analysis");
            o.bm25.unicode = flags & 1;
            o.bm25.stop_words = flags & 2;
            o.bm25.stem = flags & 4;
        }
    }
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
    c->options_.schema = c->metadata_.current_schema();  // the schema in force, as stored (options() reported it as dynamic before)
    try {  // stored settings must suit the stored schema (a strict one needs the reserved fields)
        c->check_runtime(c->options_.ttl, c->options_.string_ids);
    } catch (const std::invalid_argument& e) {
        throw FormatError(std::string("collection: settings do not suit the schema: ") + e.what());
    }
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

// ---------------------------------------------------------------- expiry and string keys

std::optional<Filter> Collection::alive_filter() const {
    if (!options_.ttl.enabled) return std::nullopt;
    return !Filter::lt(kExpiresField, now_ms());  // documents without an expiry time match too
}

bool Collection::expired(NodeId slot) const {
    if (!options_.ttl.enabled) return false;
    const std::optional<Value> v = metadata_.read(slot).get(kExpiresField);
    return v && v->type == FieldType::Int && v->i < now_ms();
}

void Collection::fill_reserved(Metadata& m, std::optional<std::string>& key, std::optional<std::int64_t>* expires) const {
    // Reserved fields are reported as fields of their own, not as metadata.
    if (const auto v = m.get(kKeyField); v && v->type == FieldType::Keyword) key = v->s;
    if (const auto v = m.get(kExpiresField); v && v->type == FieldType::Int && expires) *expires = v->i;
    Metadata clean;
    for (const auto& [name, value] : m.fields())
        if (value && name != kKeyField && name != kExpiresField) clean.set(name, *value);
    m = std::move(clean);
}

std::size_t Collection::remove_expired() {
    if (!runtime().ttl.enabled) return 0;
    return remove_where(Filter::lt(kExpiresField, now_ms()));
}

std::uint64_t Collection::id_of(std::string_view key) {
    std::uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a, then a SplitMix64 finish to spread the bits
    for (const char c : key) h = (h ^ static_cast<unsigned char>(c)) * 0x100000001b3ull;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

std::uint64_t Collection::keyed_id(std::string_view key, const Runtime& rt) const {
    if (!rt.string_ids) throw std::logic_error("Collection: string keys need CollectionOptions::string_ids");
    return id_of(key);
}

void Collection::add(std::string_view key, const Document& document) {
    const Runtime rt = runtime();
    const std::uint64_t id = keyed_id(key, rt);
    Prepared p = prepare_keyed(document, key, rt);
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    if (slot_of_.count(id)) {
        const auto v = metadata_.read(slot_of_.at(id)).get(kKeyField);
        // GCOVR_EXCL_BR_START: the first branch needs two different keys with the same 64-bit hash, which no test can construct
        throw std::invalid_argument(v && v->s != key ? "Collection: key \"" + std::string(key) + "\" has the same id as key \"" + v->s + "\""
                                                     : "Collection: duplicate key \"" + std::string(key) + "\"");
        // GCOVR_EXCL_BR_STOP
    }
    metadata_.validate(p.metadata);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocument, [&] {
        return walop::encode_document(id, p.parts, p.vector, p.text, p.sparse, p.metadata);
    }));
    (void)apply_upsert(id, p);
}

bool Collection::upsert(std::string_view key, const Document& document) {
    const Runtime rt = runtime();
    const std::uint64_t id = keyed_id(key, rt);
    Prepared p = prepare_keyed(document, key, rt);
    const auto lock = write_lock(mutex_, this);
    check_changeable();
    if (const auto it = slot_of_.find(id); it != slot_of_.end()) {
        const auto v = metadata_.read(it->second).get(kKeyField);
        if (v && v->s != key) throw std::invalid_argument("Collection: key \"" + std::string(key) + "\" has the same id as key \"" + v->s + "\"");  // GCOVR_EXCL_BR_LINE: needs a 64-bit hash collision between two keys
    }
    metadata_.validate(p.metadata);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kDocument, [&] {
        return walop::encode_document(id, p.parts, p.vector, p.text, p.sparse, p.metadata);
    }));
    return apply_upsert(id, p);
}

void Collection::check_runtime(const TtlOptions& ttl, bool string_ids) const {
    if (ttl.default_ms < 0) throw std::invalid_argument("TtlOptions::default_ms must not be negative");
    if (options_.schema.mode() != SchemaMode::Strict) return;
    auto declared = [&](const char* name) {
        for (const FieldSpec& f : options_.schema.fields())
            if (f.name == name) return true;
        return false;
    };
    if (ttl.enabled && !declared(kExpiresField))
        throw std::invalid_argument("Collection: expiry needs the field _expires_at (Int) in this strict schema; declare it when creating the collection");
    if (string_ids && !declared(kKeyField))
        throw std::invalid_argument("Collection: string keys need the field _key (Keyword) in this strict schema; declare it when creating the collection");
}

void Collection::change_settings(const Runtime& next) {
    const auto lock = write_lock(mutex_, this);
    check_runtime(next.ttl, next.string_ids);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kCollectionSettings, [&] {
        return walop::encode_collection_settings(next.ttl.enabled, next.ttl.default_ms, next.string_ids);
    }));
    options_.ttl = next.ttl;
    options_.string_ids = next.string_ids;
}

void Collection::set_ttl(const TtlOptions& ttl) {
    const auto lock = write_lock(mutex_, this);  // the check reads the other setting: under the lock too
    check_runtime(ttl, options_.string_ids);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kCollectionSettings, [&] {
        return walop::encode_collection_settings(ttl.enabled, ttl.default_ms, options_.string_ids);
    }));
    options_.ttl = ttl;
}

void Collection::set_string_ids(bool on) {
    const auto lock = write_lock(mutex_, this);
    check_runtime(options_.ttl, on);
    const ChangeLog::Scope logged(log_, log_.begin(walop::kCollectionSettings, [&] {
        return walop::encode_collection_settings(options_.ttl.enabled, options_.ttl.default_ms, on);
    }));
    options_.string_ids = on;
}

bool Collection::set_metadata(std::string_view key, const Metadata& metadata) {
    const std::uint64_t id = keyed_id(key, runtime());
    if (!get(key)) return false;  // absent, expired, or another key's document
    return set_metadata(id, metadata);
}

bool Collection::remove(std::string_view key) {
    const std::uint64_t id = keyed_id(key, runtime());
    {
        const auto lock = read_lock(mutex_, this);
        const auto it = slot_of_.find(id);
        if (it == slot_of_.end()) return false;
        const auto v = metadata_.read(it->second).get(kKeyField);
        if (!v || v->s != key) return false;  // another key's document: never removed by mistake
    }
    return remove(id);
}

bool Collection::contains(std::string_view key) const { return get(key).has_value(); }

std::optional<DocumentRecord> Collection::get(std::string_view key) const {
    std::optional<DocumentRecord> r = get(keyed_id(key, runtime()));
    if (r && r->key != std::optional<std::string>(std::string(key))) return std::nullopt;
    return r;
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
    if (runtime().ttl.enabled) {
        const std::size_t gone = remove_expired();
        if (gone) r.notes.push_back("removed " + std::to_string(gone) + " expired documents");
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