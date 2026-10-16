/**
 * @file flat_index.cpp
 * @brief Implementation of FlatIndex (see flat_index.h for the design).
 */
#include "flat_index.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vecdb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric, Schema schema, QuantizationParams quantization)
    : input_dim_(dim),
      metric_(metric),
      distance_(get_distance(metric)),
      vectors_(vecdb::indexed_dim(quantization, dim)),  // checks the truncation
      metadata_(std::move(schema)),
      scratch_(vecdb::indexed_dim(quantization, dim)) {
    validate(quantization);
    quant_ = quantization;
    if (metric == Metric::Cosine || vectors_.dim() < dim) originals_.emplace(dim);  // keep the vectors as given
    codec_ = make_codec(quant_.codec, vectors_.dim(), metric);
}

void FlatIndex::add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    const auto guard = lock_.write("add");
    add_unlocked(id, vector, metadata);
}

void FlatIndex::add_unlocked(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    // 1. Validate everything before changing anything.
    prepare_truncated(scratch_, vector, dim(), metric_);
    metadata_.validate(metadata);
    if (ids_.find(id)) throw std::invalid_argument("external id already exists");

    // 2. Metadata, ID, vector: each all-or-nothing, earlier steps undone on failure.
    const auto slot = static_cast<NodeId>(vectors_.size());
    const auto undo = metadata_.write(slot, metadata);
    try {
        ids_.add(id);
    } catch (...) {
        metadata_.undo_write(slot, undo);
        throw;
    }
    try {
        add_vector(vector);
    } catch (...) {
        ids_.undo_last_add(id);
        metadata_.undo_write(slot, undo);
        throw;
    }
    ++live_;
    maybe_train();
}

void FlatIndex::add_vector(std::span<const float> original) {
    const bool coded = codec_ && codec_->trained();
    if (coded) {  // room for the code first: may throw, before any change
        const std::size_t needed = (vectors_.size() + 1) * codec_->code_size();
        if (codes_.size() < needed) codes_.resize(std::max(needed, codes_.size() + codes_.size() / 2));
    }
    vectors_.add(scratch_.values());
    if (!originals_) {
        if (coded) codec_->encode(scratch_.values(), codes_.data() + (vectors_.size() - 1) * codec_->code_size());
        return;
    }
    try {
        originals_->add(original);
    } catch (...) {
        vectors_.undo_last_add();
        throw;
    }
    if (coded) codec_->encode(scratch_.values(), codes_.data() + (vectors_.size() - 1) * codec_->code_size());
}

bool FlatIndex::remove(std::uint64_t id) {
    const auto guard = lock_.write("remove");
    return remove_unlocked(id);
}

Record FlatIndex::record(NodeId slot, bool with_vector, bool with_metadata) const {
    Record r;
    r.id = ids_.external(slot);
    if (with_vector) {
        const auto v = original(slot);
        r.vector.assign(v.begin(), v.end());
    }
    if (with_metadata) r.metadata = metadata_.read(slot);
    return r;
}

std::optional<std::vector<SearchResult>> FlatIndex::search_by_id(std::uint64_t id, std::size_t k, const SearchOptions& options,
                                                                 bool include_self) const {
    const auto guard = lock_.read();  // the search below nests inside this read: the vector cannot change
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    const std::span<const float> v = original(*node);  // normalized again by search, exactly as when stored
    if (include_self || k == 0) return search(v, k, options);
    auto results = search(v, k == SIZE_MAX ? k : k + 1, options);  // one more, in case the record itself is among them
    std::erase_if(results, [&](const SearchResult& r) { return r.id == id; });
    if (results.size() > k) results.resize(k);
    return results;
}

std::size_t FlatIndex::remove_batch(std::span<const std::uint64_t> ids) {
    const auto guard = lock_.write("remove_batch");
    std::size_t removed = 0;
    for (std::uint64_t id : ids) removed += remove_unlocked(id) ? 1 : 0;  // unknown or already removed: skipped
    return removed;
}

void FlatIndex::maybe_train() noexcept {
    if (!codec_ || codec_->trained() || quant_.train_after == 0 || live_ < quant_.train_after) return;
    try {
        train_codec_unlocked();
    } catch (...) {  // out of memory: stay untrained (searches use full precision) and try again on a later add
    }
}

void FlatIndex::train_codec_unlocked() {
    if (!codec_) throw std::logic_error("train_codec: this index has no codec (see QuantizationParams)");
    if (live_ == 0) throw std::invalid_argument("train_codec: the index is empty");
    const std::size_t n = vectors_.size(), take = std::min<std::size_t>(n, 100000), d = indexed_dim();
    std::vector<float> sample(take * d);  // a deterministic sample, evenly spaced
    for (std::size_t i = 0; i < take; ++i) {
        const auto v = vectors_.get(static_cast<NodeId>(i * n / take));
        std::copy(v.begin(), v.end(), sample.begin() + std::ptrdiff_t(i * d));
    }
    auto trained = make_codec(codec_->kind(), d, metric_);
    trained->train(sample);
    std::vector<std::uint8_t> codes(n * trained->code_size());
    for (NodeId r = 0; r < n; ++r) trained->encode(vectors_.get(r), codes.data() + std::size_t(r) * trained->code_size());
    codec_ = std::move(trained);  // commit: moves only
    codes_ = std::move(codes);
}

void FlatIndex::train_codec() {
    const auto guard = lock_.write("train_codec");
    train_codec_unlocked();
}

bool FlatIndex::codec_trained() const {
    const auto guard = lock_.read();
    return codec_ && codec_->trained();
}

std::optional<std::vector<std::uint8_t>> FlatIndex::get_code(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node || !codec_ || !codec_->trained()) return std::nullopt;
    const std::uint8_t* c = codes_.data() + std::size_t(*node) * codec_->code_size();
    return std::vector<std::uint8_t>(c, c + codec_->code_size());
}

std::optional<std::vector<float>> FlatIndex::get_vector(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    const auto v = original(*node);
    return std::vector<float>(v.begin(), v.end());
}

std::optional<Record> FlatIndex::get(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    return record(*node, true, true);
}

Page FlatIndex::scan(const ScanOptions& options) const {
    const auto guard = lock_.read();
    if (options.limit == 0) throw std::invalid_argument("scan: limit must be at least 1");
    std::optional<CompiledFilter> compiled;
    if (options.filter) compiled.emplace(*options.filter, metadata_);  // throws for an unknown field under a strict schema
    // Every candidate after the cursor; then the `limit` smallest IDs, in order.
    std::vector<std::pair<std::uint64_t, NodeId>> candidates;
    for (NodeId n = 0; n < vectors_.size(); ++n) {
        const std::uint64_t id = ids_.external(n);
        if (options.after && id <= *options.after) continue;
        if (compiled && !compiled->matches(n)) continue;
        candidates.emplace_back(id, n);
    }
    const bool more = candidates.size() > options.limit;
    if (more) {
        std::nth_element(candidates.begin(), candidates.begin() + std::ptrdiff_t(options.limit), candidates.end());
        candidates.resize(options.limit);
    }
    std::sort(candidates.begin(), candidates.end());
    Page page;
    page.records.reserve(candidates.size());
    for (const auto& [id, n] : candidates) page.records.push_back(record(n, options.with_vectors, options.with_metadata));
    if (more) page.next = page.records.back().id;
    return page;
}

std::vector<NodeId> FlatIndex::matching_slots(const Filter& filter) const {
    const CompiledFilter compiled(filter, metadata_);  // throws for an unknown field under a strict schema
    std::vector<NodeId> slots;
    for (NodeId n = 0; n < vectors_.size(); ++n)
        if (compiled.matches(n)) slots.push_back(n);
    return slots;
}

std::size_t FlatIndex::count_where(const Filter& filter) const {
    const auto guard = lock_.read();
    return matching_slots(filter).size();
}

std::size_t FlatIndex::remove_where(const Filter& filter) {
    const auto guard = lock_.write("remove_where");
    // Collect the user IDs first: removing swaps the last vector into the freed
    // slot, so slot numbers change as we go. After this, nothing allocates.
    const std::vector<NodeId> slots = matching_slots(filter);
    std::vector<std::uint64_t> ids;
    ids.reserve(slots.size());
    for (NodeId n : slots) ids.push_back(ids_.external(n));
    for (std::uint64_t id : ids) remove_unlocked(id);
    return ids.size();
}

std::size_t FlatIndex::set_metadata_where(const Filter& filter, const Metadata& changes) {
    const auto guard = lock_.write("set_metadata_where");
    metadata_.validate(changes);  // rejected even when nothing matches
    const std::vector<NodeId> slots = matching_slots(filter);
    if (slots.empty()) return 0;
    MetadataStore staged = metadata_;  // all or nothing: change a copy, swap it in when complete
    for (NodeId n : slots) staged.update(n, changes);
    metadata_ = std::move(staged);
    return slots.size();
}

bool FlatIndex::remove_unlocked(std::uint64_t id) {
    const auto node = ids_.find(id);
    if (!node) return false;
    // Swap-with-last: move the last vector (and its metadata) into the freed slot,
    // then shrink by one. Every step below is non-allocating and cannot throw.
    const auto last = static_cast<NodeId>(vectors_.size() - 1);
    ids_.release(id);
    if (*node != last) {
        vectors_.move_row(last, *node);
        if (originals_) originals_->move_row(last, *node);
        if (codec_ && codec_->trained())
            std::copy_n(codes_.data() + std::size_t(last) * codec_->code_size(), codec_->code_size(),
                        codes_.data() + std::size_t(*node) * codec_->code_size());
        ids_.move_slot(last, *node);
        metadata_.move_row(last, *node);
    } else {
        metadata_.clear(last);
    }
    ids_.pop_back_slot();
    vectors_.pop_back();
    if (originals_) originals_->pop_back();
    --live_;
    return true;
}

void FlatIndex::add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads,
                          const OperationOptions& options) {
    (void)threads;  // Flat inserts are copies, done on the calling thread
    const auto guard = lock_.write("add_batch");
    add_batch_unlocked(ids, vectors, nullptr, options);
}

void FlatIndex::add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                          std::span<const Metadata> metadata, std::size_t threads, const OperationOptions& options) {
    (void)threads;
    if (metadata.size() != ids.size()) throw std::invalid_argument("add_batch: one metadata entry per vector is required");
    const auto guard = lock_.write("add_batch");
    add_batch_unlocked(ids, vectors, metadata.data(), options);
}

void FlatIndex::add_batch_unlocked(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                                   const Metadata* metadata, const OperationOptions& options) {
    const std::size_t n = ids.size(), d = dim();
    if (vectors.size() != n * d) throw std::invalid_argument("add_batch: expected ids.size() * dim() values");
    if (n == 0) return;
    const Metadata none;  // for batches without metadata
    auto row = [&](std::size_t i) -> const Metadata& { return metadata ? metadata[i] : none; };

    // 1. Validate everything before changing anything.
    {
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            prepare_truncated(scratch_, vectors.subspan(i * d, d), dim(), metric_);
            if (ids_.find(ids[i]) || !seen.insert(ids[i]).second)
                throw std::invalid_argument("add_batch: an ID is already stored or appears twice in the batch");
        }
        if (metadata) metadata_.validate_batch(metadata, n);
    }

    // 2. Add in order; if memory runs out, take every added vector out again,
    //    newest first, which restores the index exactly.
    std::vector<MetadataStore::WriteUndo> undo;
    undo.reserve(n);
    detail::OperationControl control(options, n);  // may throw Cancelled before any change
    try {
        for (std::size_t i = 0; i < n; ++i) {
            prepare_truncated(scratch_, vectors.subspan(i * d, d), dim(), metric_);
            const auto slot = static_cast<NodeId>(vectors_.size());
            undo.push_back(metadata_.write(slot, row(i)));
            try {
                ids_.add(ids[i]);
            } catch (...) {
                metadata_.undo_write(slot, undo.back());
                undo.pop_back();
                throw;
            }
            try {
                add_vector(vectors.subspan(i * d, d));
            } catch (...) {
                ids_.undo_last_add(ids[i]);
                metadata_.undo_write(slot, undo.back());
                undo.pop_back();
                throw;
            }
            ++live_;
            control.advance();  // may throw Cancelled: everything added is taken out again
        }
        control.finish();  // inside: a throwing progress callback undoes the batch too
    } catch (...) {
        for (std::size_t k = undo.size(); k-- > 0;) {
            const auto slot = static_cast<NodeId>(vectors_.size() - 1);
            vectors_.pop_back();
            if (originals_) originals_->pop_back();
            ids_.undo_last_add(ids[k]);
            metadata_.undo_write(slot, undo[k]);
            --live_;
        }
        throw;
    }
    maybe_train();
}

bool FlatIndex::update(std::uint64_t id, std::span<const float> vector) {
    const auto guard = lock_.write("update");
    return replace(id, vector, nullptr);
}

bool FlatIndex::update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    const auto guard = lock_.write("update");
    return replace(id, vector, &metadata);
}

bool FlatIndex::upsert(std::uint64_t id, std::span<const float> vector) {
    const auto guard = lock_.write("upsert");  // one lock across both steps: atomic
    if (replace(id, vector, nullptr)) return false;
    add_unlocked(id, vector, {});
    return true;
}

bool FlatIndex::upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    const auto guard = lock_.write("upsert");  // one lock across both steps: atomic
    if (replace(id, vector, &metadata)) return false;
    add_unlocked(id, vector, metadata);
    return true;
}

bool FlatIndex::replace(std::uint64_t id, std::span<const float> vector, const Metadata* metadata) {
    prepare_truncated(scratch_, vector, dim(), metric_);  // validates; may throw before any change
    if (metadata) metadata_.validate(*metadata);
    const auto node = ids_.find(id);
    if (!node) return false;
    if (metadata) metadata_.write(*node, *metadata);  // all-or-nothing
    vectors_.overwrite(*node, scratch_.values());     // cannot fail: already validated
    if (originals_) originals_->overwrite(*node, vector);
    if (codec_ && codec_->trained()) codec_->encode(scratch_.values(), codes_.data() + std::size_t(*node) * codec_->code_size());
    return true;
}

bool FlatIndex::contains(std::uint64_t id) const {
    const auto guard = lock_.read();
    return ids_.find(id).has_value();
}

bool FlatIndex::set_metadata(std::uint64_t id, const Metadata& metadata) {
    const auto guard = lock_.write("set_metadata");
    const auto node = ids_.find(id);
    if (!node) return false;
    metadata_.update(*node, metadata);
    return true;
}

std::optional<Metadata> FlatIndex::get_metadata(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    return metadata_.read(*node);
}

Eligibility FlatIndex::eligibility(const SearchOptions& options, std::optional<CompiledFilter>& holder) const {
    if (options.planner) validate(*options.planner);
    if (options.filter && !options.filter->is_all()) holder.emplace(*options.filter, metadata_);
    return Eligibility{&ids_, holder ? &*holder : nullptr, options.predicate ? &options.predicate : nullptr};
}

std::vector<SearchResult> FlatIndex::scan(const float* query, std::size_t k, const Eligibility& eligible,
                                          SearchStats* stats) const {
    SearchStats st;
    st.strategy = Strategy::ForceExact;
    st.reason = eligible.filtered() ? PlanReason::FlatIndex : PlanReason::NoFilter;
    std::vector<SearchResult> results;
    if (k > 0 && live_ > 0) {
        k = std::min(k, live_);  // asking for more than exists returns everything
        const bool coded = codec_ && codec_->trained();
        const bool rerank = coded && quant_.rerank;
        TopK top(rerank ? std::min(live_, k * quant_.rerank_factor) : k);
        std::vector<float> table;
        if (coded) codec_->prepare(std::span<const float>(query, indexed_dim()), table);
        const std::size_t count = vectors_.size(), stride = vectors_.stride();
        std::size_t matches = 0;
        for (NodeId node = 0; node < count; ++node) {
            if ((node & 255) == 255 && detail::search_should_stop()) break;  // deadline or cancel: best so far
            if (eligible.filtered() && !eligible(node)) continue;
            ++matches;
            const float d = coded ? codec_->distance(table, codes_.data() + std::size_t(node) * codec_->code_size())
                                  : index_distance(distance_(query, vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine);
            top.push({d, node});
        }
        st.nodes_visited = count;
        st.distance_computations = matches;
        st.estimated_matches = matches;
        st.exact_count = true;
        st.selectivity = double(matches) / double(live_);
        results.reserve(k);
        std::vector<Candidate> found = top.take_sorted();
        if (rerank) {  // the best by code, re-ranked at full precision
            for (Candidate& c : found)
                c.distance = index_distance(distance_(query, vectors_.get_padded(c.id).data(), stride), metric_ == Metric::Cosine);
            std::sort(found.begin(), found.end());
            if (found.size() > k) found.resize(k);
        }
        for (const Candidate& c : found) results.push_back({ids_.external(c.id), c.distance});
    }
    if (stats) *stats = st;
    return results;
}

std::vector<SearchResult> FlatIndex::search(std::span<const float> query, std::size_t k) const {
    return search(query, k, SearchOptions{});
}

std::vector<SearchResult> FlatIndex::search(std::span<const float> query, std::size_t k,
                                            const SearchOptions& options) const {
    const auto guard = lock_.read();
    const detail::SearchStopScope stop(options, options.stats);  // records stopped_early last
    PreparedVector q(indexed_dim());
    prepare_truncated(q, query, dim(), metric_);  // validates even when the index is empty
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    return scan(q.data(), k, eligible, options.stats);
}

std::vector<std::vector<SearchResult>> FlatIndex::search_batch(std::span<const float> queries, std::size_t k,
                                                               std::size_t threads,
                                                               const SearchOptions& options) const {
    const auto guard = lock_.read();
    const std::size_t nq = validate_batch(queries, dim());  // every query checked before any work
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    std::vector<std::vector<SearchResult>> results(nq);
    if (nq == 0 || k == 0 || live_ == 0) return results;
    k = std::min(k, live_);
    if (codec_ && codec_->trained()) {  // codes: each query scans the codes (the tiles below are for full precision)
        std::atomic<bool> stopped{false};
        auto one = [&](std::size_t q) {
            const detail::SearchStopScope stop(options, nullptr);
            PreparedVector pq(indexed_dim());
            prepare_truncated(pq, queries.subspan(q * dim(), dim()), dim(), metric_);
            results[q] = scan(pq.data(), k, eligible, nullptr);
            if (stop.stopped()) stopped = true;
        };
        const std::size_t workers = std::min(effective_threads(threads), nq);
        if (workers <= 1) {
            for (std::size_t q = 0; q < nq; ++q) one(q);
        } else {
            pool_.get(workers)->parallel_for(nq, one);
        }
        if (options.stats) options.stats->stopped_early = stopped.load();
        return results;
    }

    // Tiling: a block of stored vectors is compared with a chunk of queries, so
    // each vector loaded into the cache serves every query in the chunk. Rows are
    // pushed in increasing order for every query, exactly as in search().
    constexpr std::size_t kQueryChunk = 16, kRowBlock = 256;
    const std::size_t chunks = (nq + kQueryChunk - 1) / kQueryChunk;
    const std::size_t count = vectors_.size(), stride = vectors_.stride();
    std::atomic<bool> any_stopped{false};
    auto run_chunk = [&](std::size_t chunk) {
        const detail::SearchStopScope stop(options, nullptr);
        struct Note {
            const detail::SearchStopScope& s;
            std::atomic<bool>& any;
            ~Note() { if (s.stopped()) any = true; }
        } note{stop, any_stopped};
        const IndexLock::Hold inside(lock_);  // pool threads work under the caller's read
        const std::size_t q0 = chunk * kQueryChunk, q1 = std::min(nq, q0 + kQueryChunk);
        std::vector<PreparedVector> prepared;
        prepared.reserve(q1 - q0);
        std::vector<TopK> tops;
        tops.reserve(q1 - q0);
        for (std::size_t q = q0; q < q1; ++q) {
            prepared.emplace_back(indexed_dim());
            prepare_truncated(prepared.back(), queries.subspan(q * dim(), dim()), dim(), metric_);
            tops.emplace_back(k);
        }
        std::vector<std::uint8_t> ok(kRowBlock);
        for (std::size_t r0 = 0; r0 < count; r0 += kRowBlock) {
            if (detail::search_should_stop()) break;  // deadline or cancel: best so far
            const std::size_t r1 = std::min(count, r0 + kRowBlock);
            for (std::size_t r = r0; r < r1; ++r) ok[r - r0] = !eligible.filtered() || eligible(NodeId(r));
            for (std::size_t q = 0; q < tops.size(); ++q)
                for (std::size_t r = r0; r < r1; ++r) {
                    if (!ok[r - r0]) continue;
                    const auto node = static_cast<NodeId>(r);
                    tops[q].push({index_distance(distance_(prepared[q].data(), vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine), node});
                }
        }
        for (std::size_t q = 0; q < tops.size(); ++q)
            for (const Candidate& c : tops[q].take_sorted()) results[q0 + q].push_back({ids_.external(c.id), c.distance});
    };
    if (threads == 1 || chunks == 1) {
        for (std::size_t c = 0; c < chunks; ++c) run_chunk(c);
    } else {
        pool_.get(threads)->parallel_for(chunks, run_chunk);
    }
    if (options.stats) options.stats->stopped_early = any_stopped.load();
    return results;
}

std::vector<SearchResult> FlatIndex::search_range(std::span<const float> query, float radius,
                                                  std::size_t max_results, const SearchOptions& options) const {
    const auto guard = lock_.read();
    const detail::SearchStopScope stop(options, options.stats);
    PreparedVector q(indexed_dim());
    prepare_truncated(q, query, dim(), metric_);
    if (std::isnan(radius)) throw std::invalid_argument("radius is NaN");
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    SearchStats st;
    st.strategy = Strategy::ForceExact;
    st.reason = PlanReason::FlatIndex;
    std::vector<Candidate> hits;
    if (max_results > 0 && live_ > 0) {
        const std::size_t count = vectors_.size(), stride = vectors_.stride();
        for (NodeId node = 0; node < count; ++node) {
            if ((node & 255) == 255 && detail::search_should_stop()) break;
            if (eligible.filtered() && !eligible(node)) continue;
            ++st.distance_computations;
            const float d = index_distance(distance_(q.data(), vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine);
            if (d <= radius) hits.push_back({d, node});
        }
        st.nodes_visited = count;
        std::sort(hits.begin(), hits.end());
        if (hits.size() > max_results) hits.resize(max_results);
    }
    if (options.stats) *options.stats = st;
    std::vector<SearchResult> results;
    results.reserve(hits.size());
    for (const Candidate& c : hits) results.push_back({ids_.external(c.id), c.distance});
    return results;
}

}  // namespace vecdb