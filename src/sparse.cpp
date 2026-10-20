/**
 * @file sparse.cpp
 * @brief Sparse vectors, the inverted index, BM25 and fusion; see sparse.h.
 *
 * Removal is lazy: a removed document leaves its posting entries in place
 * (searches skip dead slots) until dead entries outnumber live ones, when the
 * posting lists are rebuilt and slots renumbered. Document frequencies are kept
 * exact at every step.
 */
#include "sparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

#include "serialize.h"

namespace vecdb {

namespace {

constexpr std::uint32_t kDocs = persist::tag("SDOC");
constexpr std::uint32_t kMeta = persist::tag("SMET");

}  // namespace

double sparse_dot(const SparseVector& a, const SparseVector& b) {
    std::unordered_map<std::uint32_t, float> small;
    const SparseVector& s = a.size() <= b.size() ? a : b;
    const SparseVector& l = a.size() <= b.size() ? b : a;
    for (std::size_t i = 0; i < s.size(); ++i) small[s.indices[i]] += s.values[i];
    double dot = 0;
    for (std::size_t i = 0; i < l.size(); ++i)
        if (const auto it = small.find(l.indices[i]); it != small.end()) dot += double(it->second) * l.values[i];
    return dot;
}

// ---------------------------------------------------------------- SparseIndex

SparseIndex::SparseIndex(Schema schema) : metadata_(std::move(schema)) {}

SparseVector SparseIndex::normalized(const SparseVector& v) {
    if (v.indices.size() != v.values.size()) throw std::invalid_argument("sparse vector: indices and values differ in length");
    std::vector<std::size_t> order(v.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return v.indices[x] < v.indices[y]; });
    SparseVector out;
    out.indices.reserve(v.size());
    out.values.reserve(v.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        const std::size_t j = order[i];
        if (!std::isfinite(v.values[j])) throw std::invalid_argument("sparse vector: weight is NaN or infinite");
        if (i > 0 && v.indices[j] == v.indices[order[i - 1]]) throw std::invalid_argument("sparse vector: repeated term");
        if (v.values[j] == 0.0f) continue;
        out.indices.push_back(v.indices[j]);
        out.values.push_back(v.values[j]);
    }
    return out;
}

void SparseIndex::insert_unlocked(std::uint64_t id, SparseVector v, const Metadata* metadata, bool keep_metadata) {
    // keep_metadata: a replacement keeping the old row (update without metadata).
    const auto old = slot_of_.find(id);
    const bool replacing = old != slot_of_.end();
    Metadata kept;
    if (keep_metadata && replacing) kept = metadata_.read(old->second);
    const Metadata& md = metadata ? *metadata : kept;
    metadata_.validate(md);
    if (id_of_.size() >= std::size_t(kEmpty)) throw std::length_error("sparse index: too many slots (compact it)");  // GCOVR_EXCL_LINE: 4 billion slots
    const NodeId slot = static_cast<NodeId>(id_of_.size());
    // Every allocation first: once the metadata is written, nothing can fail.
    id_of_.reserve(id_of_.size() + 1);
    live_.reserve(live_.size() + 1);
    docs_.reserve(docs_.size() + 1);
    for (const std::uint32_t t : v.indices) {
        Posting& p = postings_[t];
        p.slots.reserve(p.slots.size() + 1);
        p.weights.reserve(p.weights.size() + 1);
        df_.try_emplace(t, 0);  // incremented below without allocating
    }
    auto unreserve = [&]() noexcept {  // drop the counts created above if this fails
        for (const std::uint32_t t : v.indices)
            if (const auto d = df_.find(t); d != df_.end() && d->second == 0) df_.erase(d);
    };
    std::optional<MetadataStore::WriteUndo> undo;
    std::unordered_map<std::uint64_t, NodeId>::iterator at;
    try {
        undo = metadata_.write(slot, md);
        at = replacing ? old : slot_of_.try_emplace(id, slot).first;
    } catch (...) {
        if (undo) metadata_.undo_write(slot, *undo);
        unreserve();
        throw;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        Posting& p = postings_[v.indices[i]];
        p.slots.push_back(slot);
        p.weights.push_back(v.values[i]);
        ++df_[v.indices[i]];
    }
    posting_entries_ += v.size();
    if (replacing) {  // retire the old version (its terms are counted down after the new ones are in)
        const NodeId was = at->second;
        for (const std::uint32_t t : docs_[was].indices)
            if (--df_[t] == 0) df_.erase(t);
        dead_entries_ += docs_[was].size();
        docs_[was] = {};
        live_[was] = 0;
        metadata_.clear(was);
        --live_count_;
    }
    at->second = slot;
    id_of_.push_back(id);
    live_.push_back(1);
    docs_.push_back(std::move(v));
    ++live_count_;
}

void SparseIndex::add(std::uint64_t id, const SparseVector& vector, const Metadata& metadata) {
    SparseVector v = normalized(vector);
    const auto guard = lock_.write("add");
    if (slot_of_.count(id)) throw std::invalid_argument("duplicate id " + std::to_string(id));
    insert_unlocked(id, std::move(v), &metadata, false);
}

bool SparseIndex::remove_unlocked(std::uint64_t id) {
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return false;
    const NodeId slot = it->second;
    for (const std::uint32_t t : docs_[slot].indices)
        if (--df_[t] == 0) df_.erase(t);
    dead_entries_ += docs_[slot].size();
    docs_[slot] = {};
    live_[slot] = 0;
    metadata_.clear(slot);
    slot_of_.erase(it);
    --live_count_;
    return true;
}

bool SparseIndex::remove(std::uint64_t id) {
    const auto guard = lock_.write("remove");
    const bool removed = remove_unlocked(id);
    maybe_compact();
    return removed;
}

std::size_t SparseIndex::remove_batch(std::span<const std::uint64_t> ids) {
    const auto guard = lock_.write("remove_batch");
    std::size_t n = 0;
    for (const std::uint64_t id : ids) n += remove_unlocked(id);
    maybe_compact();
    return n;
}

std::size_t SparseIndex::remove_where(const Filter& filter) {
    const auto guard = lock_.write("remove_where");
    const CompiledFilter compiled(filter, metadata_);
    std::vector<std::uint64_t> ids;
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n] && compiled.matches(n)) ids.push_back(id_of_[n]);
    for (const std::uint64_t id : ids) (void)remove_unlocked(id);
    maybe_compact();
    return ids.size();
}

bool SparseIndex::update(std::uint64_t id, const SparseVector& vector) {
    SparseVector v = normalized(vector);
    const auto guard = lock_.write("update");
    if (!slot_of_.count(id)) return false;
    insert_unlocked(id, std::move(v), nullptr, true);
    maybe_compact();
    return true;
}

bool SparseIndex::update(std::uint64_t id, const SparseVector& vector, const Metadata& metadata) {
    SparseVector v = normalized(vector);
    const auto guard = lock_.write("update");
    if (!slot_of_.count(id)) return false;
    insert_unlocked(id, std::move(v), &metadata, false);
    maybe_compact();
    return true;
}

bool SparseIndex::upsert(std::uint64_t id, const SparseVector& vector) {
    SparseVector v = normalized(vector);
    const auto guard = lock_.write("upsert");
    const bool added = !slot_of_.count(id);
    insert_unlocked(id, std::move(v), nullptr, true);
    maybe_compact();
    return added;
}

bool SparseIndex::upsert(std::uint64_t id, const SparseVector& vector, const Metadata& metadata) {
    SparseVector v = normalized(vector);
    const auto guard = lock_.write("upsert");
    const bool added = !slot_of_.count(id);
    insert_unlocked(id, std::move(v), &metadata, false);
    maybe_compact();
    return added;
}

bool SparseIndex::set_metadata(std::uint64_t id, const Metadata& metadata) {
    const auto guard = lock_.write("set_metadata");
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return false;
    metadata_.update(it->second, metadata);
    return true;
}

bool SparseIndex::contains(std::uint64_t id) const {
    const auto guard = lock_.read();
    return slot_of_.count(id) != 0;
}

std::size_t SparseIndex::size() const {
    const auto guard = lock_.read();
    return live_count_;
}

std::optional<SparseVector> SparseIndex::get_vector(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return std::nullopt;
    return docs_[it->second];
}

std::optional<Metadata> SparseIndex::get_metadata(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto it = slot_of_.find(id);
    if (it == slot_of_.end()) return std::nullopt;
    return metadata_.read(it->second);
}

std::vector<SparseRecord> SparseIndex::records() const {
    const auto guard = lock_.read();
    std::vector<SparseRecord> out;
    out.reserve(live_count_);
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n]) out.push_back({id_of_[n], docs_[n], metadata_.read(n)});
    std::sort(out.begin(), out.end(), [](const SparseRecord& a, const SparseRecord& b) { return a.id < b.id; });
    return out;
}

std::size_t SparseIndex::document_frequency(std::uint32_t term) const {
    const auto guard = lock_.read();
    const auto it = df_.find(term);
    return it == df_.end() ? 0 : it->second;
}

std::size_t SparseIndex::terms() const {
    const auto guard = lock_.read();
    return df_.size();
}

std::size_t SparseIndex::postings() const {
    const auto guard = lock_.read();
    return posting_entries_;
}

std::size_t SparseIndex::dead_postings() const {
    const auto guard = lock_.read();
    return dead_entries_;
}

std::size_t SparseIndex::slots() const {
    const auto guard = lock_.read();
    return id_of_.size();
}

std::vector<SearchResult> SparseIndex::search(const SparseVector& query, std::size_t k, const SearchOptions& options) const {
    const SparseVector q = normalized(query);
    const auto guard = lock_.read();
    std::optional<CompiledFilter> compiled;
    if (options.filter) compiled.emplace(*options.filter, metadata_);
    if (options.stats) *options.stats = {};
    if (k == 0 || q.empty() || live_count_ == 0) return {};
    // Accumulate the products of the query's terms over their posting lists, in
    // per-thread buffers reused across searches: only the slots touched are reset,
    // so a search costs its posting lists, not the index size. A search nested in
    // a predicate on the same thread gets buffers of its own.
    struct Buffers {
        std::vector<float> score;
        std::vector<std::uint8_t> seen;
        std::vector<NodeId> touched;
        bool busy = false;
    };
    thread_local Buffers shared;
    Buffers own;
    Buffers& buf = shared.busy ? own : shared;
    if (buf.score.size() < id_of_.size()) {
        buf.score.resize(id_of_.size(), 0.0f);
        buf.seen.resize(id_of_.size(), 0);
    }
    buf.touched.clear();
    buf.busy = true;
    struct Release {
        Buffers& b;
        ~Release() {
            for (const NodeId s : b.touched) {
                b.score[s] = 0.0f;
                b.seen[s] = 0;
            }
            b.touched.clear();
            b.busy = false;
        }
    } release{buf};
    std::vector<float>& score = buf.score;
    std::vector<std::uint8_t>& seen = buf.seen;
    std::vector<NodeId>& touched = buf.touched;
    bool stopped = false;
    std::size_t visited = 0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        if ((options.deadline && std::chrono::steady_clock::now() >= *options.deadline) ||
            (options.cancel && options.cancel->load())) {
            stopped = true;
            break;
        }
        const auto it = postings_.find(q.indices[i]);
        if (it == postings_.end()) continue;
        const Posting& p = it->second;
        const float w = q.values[i];
        for (std::size_t j = 0; j < p.slots.size(); ++j) {
            const NodeId s = p.slots[j];
            if (!live_[s]) continue;
            if (!seen[s]) {
                seen[s] = 1;
                touched.push_back(s);
            }
            score[s] += w * p.weights[j];
        }
        visited += p.slots.size();
    }
    std::vector<SearchResult> found;
    found.reserve(touched.size());
    for (const NodeId s : touched) {
        if (compiled && !compiled->matches(s)) continue;
        if (options.predicate && !options.predicate(id_of_[s])) continue;
        found.push_back({id_of_[s], -score[s]});
    }
    const auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    };
    const std::size_t n = std::min(k, found.size());
    std::partial_sort(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(n), found.end(), better);
    found.resize(n);
    if (options.stats) {
        options.stats->nodes_visited = touched.size();
        options.stats->distance_computations = visited;
        options.stats->stopped_early = stopped;
    }
    return found;
}

void SparseIndex::maybe_compact() {
    // Dead posting entries slow searches; dead slots (every update leaves one, and
    // empty documents leave no entries) cost memory and could exhaust 32-bit slots.
    const std::size_t dead_slots = id_of_.size() - live_count_;
    if ((dead_entries_ > 1024 && dead_entries_ > posting_entries_ - dead_entries_) || (dead_slots > 1024 && dead_slots > live_count_)) {
        try {
            compact_unlocked();  // all or nothing
        } catch (const std::bad_alloc&) {
            // Optional: the change before it is done, and the next one retries.
        }
    }
}

void SparseIndex::compact() {
    const auto guard = lock_.write("compact");
    compact_unlocked();
}

void SparseIndex::compact_unlocked() {
    // Built aside, then swapped in: an allocation failure leaves the index as it was.
    std::vector<std::uint64_t> ids;
    std::vector<SparseVector> docs;
    std::vector<NodeId> from;
    ids.reserve(live_count_);
    docs.reserve(live_count_);
    from.reserve(live_count_);
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n]) {
            ids.push_back(id_of_[n]);
            docs.push_back(docs_[n]);
            from.push_back(n);
        }
    std::unordered_map<std::uint32_t, Posting> postings;
    std::size_t entries = 0;
    for (NodeId s = 0; s < docs.size(); ++s)
        for (std::size_t i = 0; i < docs[s].size(); ++i) {
            Posting& p = postings[docs[s].indices[i]];
            p.slots.push_back(s);
            p.weights.push_back(docs[s].values[i]);
            ++entries;
        }
    std::unordered_map<std::uint64_t, NodeId> slot_of;
    slot_of.reserve(ids.size());
    for (NodeId s = 0; s < ids.size(); ++s) slot_of.emplace(ids[s], s);
    std::vector<std::uint8_t> live(ids.size(), 1);
    // Nothing below allocates.
    for (NodeId s = 0; s < from.size(); ++s)
        if (from[s] != s) metadata_.move_row(from[s], s);
    id_of_.swap(ids);
    docs_.swap(docs);
    live_.swap(live);
    slot_of_.swap(slot_of);
    postings_.swap(postings);
    posting_entries_ = entries;
    dead_entries_ = 0;
}

// ---------------------------------------------------------------- persistence

void SparseIndex::save(std::ostream& out) const {
    const auto guard = lock_.read();
    persist::StreamErrorsThrow errors(out);
    persist::write_header(out, persist::IndexKind::Sparse);
    // Live documents only, in slot order: what loading produces (a compacted
    // index), so the bytes do not depend on how many removals await compaction.
    persist::write_section(out, kDocs, [&](persist::Writer& w) {
        w.put<std::uint64_t>(live_count_);
        for (NodeId n = 0; n < id_of_.size(); ++n) {
            if (!live_[n]) continue;
            w.put<std::uint8_t>(live_[n]);
            w.put<std::uint64_t>(id_of_[n]);
            w.put<std::uint64_t>(docs_[n].size());
            for (const std::uint32_t t : docs_[n].indices) w.put<std::uint32_t>(t);
            w.put_floats(docs_[n].values.data(), docs_[n].values.size());
        }
    });
    // Metadata rows in the same order: a copy compacted like the documents.
    MetadataStore rows = metadata_;
    NodeId to = 0;
    for (NodeId n = 0; n < id_of_.size(); ++n)
        if (live_[n]) {
            if (n != to) rows.move_row(n, to);
            ++to;
        }
    persist::write_section(out, kMeta, [&](persist::Writer& w) { rows.save(w); });
}

void SparseIndex::save(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create " + tmp);
        try {
            save(out);
        } catch (...) {
            out.close();
            std::remove(tmp.c_str());
            throw;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // GCOVR_EXCL_START: a rename onto an existing file fails on some systems
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path);
    }  // GCOVR_EXCL_STOP
}

std::unique_ptr<SparseIndex> SparseIndex::load(std::istream& in) {
    persist::Reader r(in);
    r.read_header(persist::IndexKind::Sparse);
    auto index = std::make_unique<SparseIndex>();
    SparseIndex& x = *index;
    r.begin_section(kDocs, "documents");
    const std::uint64_t slots = r.get_count(17, "documents");
    x.id_of_.reserve(slots);
    x.live_.reserve(slots);
    x.docs_.reserve(slots);
    for (std::uint64_t n = 0; n < slots; ++n) {
        const std::uint8_t live = r.get<std::uint8_t>();
        if (live > 1) throw FormatError("sparse index: invalid live flag");
        const std::uint64_t id = r.get<std::uint64_t>();
        const std::uint64_t nnz = r.get_count(8, "terms");
        SparseVector v;
        v.indices.resize(nnz);
        v.values.resize(nnz);
        for (auto& t : v.indices) t = r.get<std::uint32_t>();
        r.get_floats(v.values.data(), nnz);
        for (std::size_t i = 0; i < nnz; ++i) {
            if ((i > 0 && v.indices[i] <= v.indices[i - 1]) || !std::isfinite(v.values[i]) || v.values[i] == 0.0f)
                throw FormatError("sparse index: invalid document terms");
        }
        if (!live && nnz != 0) throw FormatError("sparse index: removed document with terms");
        if (live && !x.slot_of_.emplace(id, static_cast<NodeId>(n)).second) throw FormatError("sparse index: duplicate id");
        x.id_of_.push_back(id);
        x.live_.push_back(live);
        x.docs_.push_back(std::move(v));
    }
    r.end_section();
    r.begin_section(kMeta, "metadata");
    x.metadata_ = MetadataStore::load(r, static_cast<std::size_t>(slots), [&](NodeId n) { return x.live_[n] != 0; });
    r.end_section();
    r.expect_end_of_file();
    x.live_count_ = x.slot_of_.size();
    x.compact_unlocked();  // posting lists, densely numbered
    for (const SparseVector& v : x.docs_)
        for (const std::uint32_t t : v.indices) ++x.df_[t];
    return index;
}

std::unique_ptr<SparseIndex> SparseIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return load(in);
}

// ---------------------------------------------------------------- BM25

Bm25::Bm25(Bm25Params params) : params_(params) {
    if (!(params_.k1 >= 0) || !(params_.b >= 0 && params_.b <= 1) || !(params_.avg_doc_length > 0))
        throw std::invalid_argument("Bm25Params: k1 >= 0, 0 <= b <= 1 and avg_doc_length > 0 required");
}

std::vector<std::string> Bm25::tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (const char ch : text) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80) cur.push_back(ch);
        else if (c >= 'A' && c <= 'Z') cur.push_back(static_cast<char>(c - 'A' + 'a'));
        else if (!cur.empty()) tokens.push_back(std::move(cur)), cur.clear();
    }
    if (!cur.empty()) tokens.push_back(std::move(cur));
    return tokens;
}

std::uint32_t Bm25::term_id(std::string_view token) {
    std::uint32_t h = 2166136261u;
    for (const char c : token) {
        h ^= static_cast<unsigned char>(c);
        h *= 16777619u;
    }
    return h;
}

void Bm25::fit(std::span<const std::string_view> documents) {
    if (documents.empty()) return;
    double total = 0;
    for (const std::string_view d : documents) total += double(tokenize(d).size());
    if (total > 0) params_.avg_doc_length = total / double(documents.size());
}

SparseVector Bm25::encode_document(std::string_view text) const {
    const auto tokens = tokenize(text);
    std::map<std::uint32_t, double> tf;
    for (const auto& t : tokens) tf[term_id(t)] += 1;
    const double norm = params_.k1 * (1 - params_.b + params_.b * double(tokens.size()) / params_.avg_doc_length);
    SparseVector v;
    for (const auto& [term, f] : tf) {
        v.indices.push_back(term);
        v.values.push_back(static_cast<float>(f * (params_.k1 + 1) / (f + norm)));
    }
    return v;
}

SparseVector Bm25::encode_query(std::string_view text, const SparseIndex& index) const {
    const double n = double(index.size());
    std::map<std::uint32_t, double> weight;
    for (const auto& t : tokenize(text)) {
        const std::uint32_t term = term_id(t);
        const double df = double(index.document_frequency(term));
        if (df == 0) continue;
        weight[term] += std::log(1 + (n - df + 0.5) / (df + 0.5));
    }
    SparseVector v;
    for (const auto& [term, w] : weight) {
        v.indices.push_back(term);
        v.values.push_back(static_cast<float>(w));
    }
    return v;
}

// ---------------------------------------------------------------- fusion

std::vector<HybridResult> fuse(std::span<const SearchResult> dense, std::span<const SearchResult> sparse, std::size_t k,
                               const HybridOptions& o) {
    if (!(o.dense_weight >= 0) || !(o.sparse_weight >= 0) || !(o.rrf_k >= 0) || !std::isfinite(o.dense_weight) ||
        !std::isfinite(o.sparse_weight) || !std::isfinite(o.rrf_k))
        throw std::invalid_argument("HybridOptions: weights and rrf_k must be finite and non-negative");
    std::unordered_map<std::uint64_t, HybridResult> by_id;
    // Weighted: each list's similarities (-distance) scaled to [0, 1].
    auto scaled = [](std::span<const SearchResult> list, std::size_t i) {
        const double hi = -double(list.front().distance), lo = -double(list.back().distance);
        return hi == lo ? 1.0 : (-double(list[i].distance) - lo) / (hi - lo);
    };
    auto add = [&](std::span<const SearchResult> list, double weight, bool is_dense) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            HybridResult& r = by_id[list[i].id];
            r.id = list[i].id;
            if (is_dense) {
                if (r.dense_rank) continue;  // a repeated ID: the first (best) counts
                r.dense_rank = i + 1;
                r.dense_distance = list[i].distance;
            } else {
                if (r.sparse_rank) continue;
                r.sparse_rank = i + 1;
                r.sparse_score = -list[i].distance;
            }
            r.score += o.fusion == Fusion::Rrf ? weight / (o.rrf_k + double(i + 1)) : weight * scaled(list, i);
        }
    };
    add(dense, o.dense_weight, true);
    add(sparse, o.sparse_weight, false);
    std::vector<HybridResult> out;
    out.reserve(by_id.size());
    for (auto& [id, r] : by_id) out.push_back(r);
    const std::size_t n = std::min(k, out.size());
    std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), out.end(),
                      [](const HybridResult& a, const HybridResult& b) { return a.score > b.score || (a.score == b.score && a.id < b.id); });
    out.resize(n);
    return out;
}

}  // namespace vecdb
