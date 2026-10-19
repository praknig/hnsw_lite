/**
 * @file flat_index.cpp
 * @brief Implementation of FlatIndex (see flat_index.h for the design).
 */
#include "flat_index.h"
#include <limits>
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vecdb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric, Schema schema, QuantizationParams quantization, IvfParams ivf)
    : input_dim_(dim),
      metric_(metric),
      distance_(get_distance(metric)),
      vectors_(vecdb::indexed_dim(quantization, dim)),  // checks the truncation
      metadata_(std::move(schema)),
      scratch_(vecdb::indexed_dim(quantization, dim)) {
    validate(quantization);
    validate(ivf);
    quant_ = quantization;
    ivf_ = ivf;
    if (keeps_originals(quant_, metric, dim)) originals_.emplace(dim);  // keep the vectors as given
    codec_ = make_codec(quant_.codec, vectors_.dim(), metric, quant_.pq_subspaces);
}

void FlatIndex::add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    const auto guard = lock_.write("add");
    add_unlocked(id, vector, metadata);
}

void FlatIndex::add_unlocked(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    note_change(id);  // for a running rebuild(); harmless if nothing changes
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
    maybe_train_lists();
}

void FlatIndex::add_vector(std::span<const float> original) {
    const bool coded = codec_ && codec_->trained();
    std::uint32_t list = 0;
    if (ivf_trained()) {  // its list first, with room: may throw, before any change
        list = nearest_list(scratch_.data());
        ivf_reserve(list);
    }
    if (coded) {  // room for the code first: may throw, before any change
        const std::size_t needed = (vectors_.size() + 1) * codec_->code_size();
        if (codes_.size() < needed) codes_.resize(std::max(needed, codes_.size() + codes_.size() / 2));
    }
    vectors_.add(scratch_.values());
    if (!originals_) {
        if (coded) codec_->encode(scratch_.values(), codes_.data() + (vectors_.size() - 1) * codec_->code_size());
        if (ivf_trained()) ivf_append(static_cast<NodeId>(vectors_.size() - 1), list);
        return;
    }
    try {
        originals_->add(original);
    } catch (...) {
        vectors_.undo_last_add();
        throw;
    }
    if (coded) codec_->encode(scratch_.values(), codes_.data() + (vectors_.size() - 1) * codec_->code_size());
    if (ivf_trained()) ivf_append(static_cast<NodeId>(vectors_.size() - 1), list);
}

bool FlatIndex::remove(std::uint64_t id) {
    const auto guard = lock_.write("remove");
    return remove_unlocked(id);
}

Record FlatIndex::record(NodeId slot, bool with_vector, bool with_metadata) const {
    Record r;
    r.id = ids_.external(slot);
    if (with_vector) {
        r.vector = read_back(slot);
    }
    if (with_metadata) r.metadata = metadata_.read(slot);
    return r;
}

std::optional<std::vector<SearchResult>> FlatIndex::search_by_id(std::uint64_t id, std::size_t k, const SearchOptions& options,
                                                                 bool include_self) const {
    const auto guard = lock_.read();  // the search below nests inside this read: the vector cannot change
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    const std::vector<float> v = as_input(*node);  // prepared again by search, as when stored
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
    if (codes_only()) throw std::logic_error("train_codec: this index keeps only codes; training again from decoded vectors could only lose precision");  // see HnswIndex
    const std::size_t n = vectors_.size(), take = std::min<std::size_t>(n, 100000), d = indexed_dim();
    std::vector<float> sample(take * d);  // a deterministic sample, evenly spaced
    for (std::size_t i = 0; i < take; ++i) {
        const auto v = stored_copy(static_cast<NodeId>(i * n / take));  // decoded, with codes only
        std::copy(v.begin(), v.end(), sample.begin() + std::ptrdiff_t(i * d));
    }
    auto trained = make_codec(codec_->kind(), d, metric_, quant_.pq_subspaces);
    trained->train(sample);
    std::vector<std::uint8_t> codes(n * trained->code_size());
    for (NodeId r = 0; r < n; ++r) trained->encode(stored_copy(r), codes.data() + std::size_t(r) * trained->code_size());
    codec_ = std::move(trained);  // commit: moves only, and freeing memory
    codes_ = std::move(codes);
    if (!quant_.keep_full_vectors) vectors_.drop_values();  // keep only codes from now on
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

std::vector<float> FlatIndex::stored_copy(NodeId slot) const {
    if (!codes_only()) {
        const auto v = vectors_.get(slot);
        return std::vector<float>(v.begin(), v.end());
    }
    std::vector<float> v(indexed_dim());
    codec_->decode(codes_.data() + std::size_t(slot) * codec_->code_size(), v);
    return v;
}

std::vector<float> FlatIndex::read_back(NodeId slot) const {
    if (originals_) {
        const auto v = originals_->get(slot);
        return std::vector<float>(v.begin(), v.end());
    }
    return stored_copy(slot);
}

std::vector<float> FlatIndex::as_input(NodeId slot) const {
    std::vector<float> input = read_back(slot);
    input.resize(dim(), 0.0f);  // only when truncating without full vectors
    return input;
}

std::uint32_t FlatIndex::nearest_list(const float* padded) const { return nearest_of(centroids_, padded); }

std::uint32_t FlatIndex::nearest_of(const std::vector<float>& centroids, const float* padded) const {
    const std::size_t stride = vectors_.stride(), lists = centroids.size() / stride;
    std::uint32_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (std::uint32_t l = 0; l < lists; ++l) {
        const float d = index_distance(distance_(padded, centroids.data() + std::size_t(l) * stride, stride), metric_ == Metric::Cosine);
        if (d < best_d) {  // strict: ties keep the lower list
            best_d = d;
            best = l;
        }
    }
    return best;
}

void FlatIndex::ivf_reserve(std::uint32_t list) {
    auto grow = [](auto& v) { if (v.size() == v.capacity()) v.reserve(std::max<std::size_t>(8, v.size() * 2)); };
    grow(lists_[list]);
    grow(list_of_);
    grow(pos_);
}

void FlatIndex::ivf_append(NodeId row, std::uint32_t list) noexcept {
    lists_[list].push_back(row);  // reserved by ivf_reserve
    list_of_.push_back(list);
    pos_.push_back(static_cast<std::uint32_t>(lists_[list].size() - 1));
}

void FlatIndex::ivf_detach(NodeId row) noexcept {
    auto& members = lists_[list_of_[row]];
    const NodeId moved = members.back();  // the list's last member fills the gap
    members[pos_[row]] = moved;
    pos_[moved] = pos_[row];
    members.pop_back();
}

void FlatIndex::ivf_pop_row(NodeId row) noexcept {
    ivf_detach(row);
    list_of_.pop_back();
    pos_.pop_back();
}

void FlatIndex::maybe_train_lists() noexcept {
    if (ivf_.lists == 0 || ivf_trained() || ivf_.train_after == 0 || live_ < std::max(ivf_.train_after, ivf_.lists)) return;
    try {
        train_lists_unlocked();
    } catch (...) {  // out of memory: stay untrained (searches scan everything) and try again on a later add
    }
}

void FlatIndex::train_lists_unlocked() {
    if (ivf_.lists == 0) throw std::logic_error("train_lists: this index has no inverted file (see IvfParams)");
    if (live_ < ivf_.lists) throw std::invalid_argument("train_lists: fewer vectors than lists");
    const std::size_t n = vectors_.size(), take = std::min<std::size_t>(n, 100000), d = indexed_dim(), stride = vectors_.stride();
    std::vector<float> sample(take * d);  // a deterministic sample, evenly spaced (decoded, with codes only)
    for (std::size_t i = 0; i < take; ++i) {
        const auto v = stored_copy(static_cast<NodeId>(i * n / take));
        std::copy(v.begin(), v.end(), sample.begin() + std::ptrdiff_t(i * d));
    }
    const KMeansResult r = kmeans(sample, d, KMeansParams{std::min(ivf_.lists, take), 25, 0x1F5EED, 0});
    const std::size_t lists = r.centroids.size() / d;
    // Everything is built in local variables; the index changes only by the
    // swaps at the end, which cannot fail (all or nothing).
    std::vector<float> centroids(lists * stride, 0.0f);
    for (std::size_t l = 0; l < lists; ++l) std::copy_n(r.centroids.begin() + std::ptrdiff_t(l * d), d, centroids.begin() + std::ptrdiff_t(l * stride));
    std::vector<std::vector<NodeId>> members(lists);
    std::vector<std::uint32_t> list_of(n), pos(n);
    std::vector<float> buffer(stride, 0.0f);
    for (NodeId row = 0; row < n; ++row) {
        const auto v = stored_copy(row);
        std::copy(v.begin(), v.end(), buffer.begin());
        const std::uint32_t l = nearest_of(centroids, buffer.data());
        list_of[row] = l;
        pos[row] = static_cast<std::uint32_t>(members[l].size());
        members[l].push_back(row);
    }
    centroids_.swap(centroids);  // commit: moves only
    lists_.swap(members);
    list_of_.swap(list_of);
    pos_.swap(pos);
}

void FlatIndex::train_lists() {
    const auto guard = lock_.write("train_lists");
    train_lists_unlocked();
}

bool FlatIndex::lists_trained() const {
    const auto guard = lock_.read();
    return ivf_trained();
}

std::optional<std::uint32_t> FlatIndex::list_of(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node || !ivf_trained()) return std::nullopt;
    return list_of_[*node];
}

std::vector<std::size_t> FlatIndex::list_sizes() const {
    const auto guard = lock_.read();
    std::vector<std::size_t> sizes;
    for (const auto& l : lists_) sizes.push_back(l.size());
    return sizes;
}

std::optional<std::vector<float>> FlatIndex::get_vector(std::uint64_t id) const {
    const auto guard = lock_.read();
    const auto node = ids_.find(id);
    if (!node) return std::nullopt;
    return read_back(*node);
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
    for (NodeId n : slots) note_change(ids_.external(n));  // every write path records its IDs (harmless here)
    MetadataStore staged = metadata_;  // all or nothing: change a copy, swap it in when complete
    for (NodeId n : slots) staged.update(n, changes);
    metadata_ = std::move(staged);
    return slots.size();
}

bool FlatIndex::remove_unlocked(std::uint64_t id) {
    note_change(id);
    const auto node = ids_.find(id);
    if (!node) return false;
    // Swap-with-last: move the last vector (and its metadata) into the freed slot,
    // then shrink by one. Every step below is non-allocating and cannot throw.
    const auto last = static_cast<NodeId>(vectors_.size() - 1);
    ids_.release(id);
    if (*node != last) {
        if (ivf_trained()) {  // the removed row leaves its list; the last row takes its number
            ivf_detach(*node);
            const std::uint32_t l = list_of_[last];
            lists_[l][pos_[last]] = *node;
            list_of_[*node] = l;
            pos_[*node] = pos_[last];
        }
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
    if (ivf_trained()) {
        if (last == *node) ivf_detach(*node);
        list_of_.pop_back();
        pos_.pop_back();
    }
    ids_.pop_back_slot();
    vectors_.pop_back();
    if (originals_) originals_->pop_back();
    --live_;
    ++removals_since_upkeep_;
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
    for (std::uint64_t id : ids) note_change(id);  // for a running rebuild(), before anything changes
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
            if (ivf_trained()) ivf_pop_row(slot);
            vectors_.pop_back();
            if (originals_) originals_->pop_back();
            ids_.undo_last_add(ids[k]);
            metadata_.undo_write(slot, undo[k]);
            --live_;
        }
        throw;
    }
    maybe_train();
    maybe_train_lists();
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
    note_change(id);
    prepare_truncated(scratch_, vector, dim(), metric_);  // validates; may throw before any change
    if (metadata) metadata_.validate(*metadata);
    const auto node = ids_.find(id);
    if (!node) return false;
    if (metadata) metadata_.write(*node, *metadata);  // all-or-nothing
    std::uint32_t new_list = 0;
    if (ivf_trained()) {  // the new vector may belong to another list: room first
        new_list = nearest_list(scratch_.data());
        if (new_list != list_of_[*node]) ivf_reserve(new_list);
    }
    vectors_.overwrite(*node, scratch_.values());     // cannot fail: already validated
    if (ivf_trained() && new_list != list_of_[*node]) {
        ivf_detach(*node);
        lists_[new_list].push_back(*node);  // reserved
        list_of_[*node] = new_list;
        pos_[*node] = static_cast<std::uint32_t>(lists_[new_list].size() - 1);
    }
    if (originals_) originals_->overwrite(*node, vector);
    if (codec_ && codec_->trained()) codec_->encode(scratch_.values(), codes_.data() + std::size_t(*node) * codec_->code_size());
    ++updates_since_upkeep_;
    return true;
}

bool FlatIndex::contains(std::uint64_t id) const {
    const auto guard = lock_.read();
    return ids_.find(id).has_value();
}

bool FlatIndex::set_metadata(std::uint64_t id, const Metadata& metadata) {
    note_change(id);
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

std::vector<SearchResult> FlatIndex::scan(const float* query, std::size_t k, const Eligibility& eligible, std::size_t probes,
                                          SearchStats* stats) const {
    SearchStats st;
    st.strategy = Strategy::ForceExact;
    st.reason = eligible.filtered() ? PlanReason::FlatIndex : PlanReason::NoFilter;
    std::vector<SearchResult> results;
    if (k > 0 && live_ > 0) {
        k = std::min(k, live_);  // asking for more than exists returns everything
        const bool coded = codec_ && codec_->trained();
        const bool rerank = coded && quant_.rerank && !codes_only();  // nothing to re-rank with otherwise
        TopK top(rerank ? std::min(live_, saturating_mul(k, quant_.rerank_factor)) : k);
        std::vector<float> table;
        if (coded) codec_->prepare(std::span<const float>(query, indexed_dim()), table);
        const std::size_t count = vectors_.size(), stride = vectors_.stride();
        std::size_t matches = 0, visited = 0;
        bool stop = false;
        auto visit = [&](NodeId node) {
            if ((++visited & 255) == 0 && detail::search_should_stop()) stop = true;  // deadline or cancel: best so far
            if (eligible.filtered() && !eligible(node)) return;
            ++matches;
            const float d = coded ? codec_->distance(table, codes_.data() + std::size_t(node) * codec_->code_size())
                                  : index_distance(distance_(query, vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine);
            top.push({d, node});
        };
        if (!ivf_trained()) {
            for (NodeId node = 0; node < count && !stop; ++node) visit(node);
        } else {
            // The nearest lists first: at least `probes`, and more until k matching
            // vectors were seen (a selective filter may need more), or all of them.
            std::vector<std::pair<float, std::uint32_t>> order(lists_.size());
            for (std::uint32_t l = 0; l < lists_.size(); ++l)
                order[l] = {index_distance(distance_(query, centroids_.data() + std::size_t(l) * stride, stride), metric_ == Metric::Cosine), l};
            std::sort(order.begin(), order.end());
            const std::size_t want = probes == 0 ? ivf_.probes : probes;
            for (std::size_t p = 0; p < order.size() && !stop && (p < want || matches < k); ++p)
                for (NodeId node : lists_[order[p].second]) {
                    visit(node);
                    if (stop) break;
                }
        }
        st.nodes_visited = visited;
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
    return scan(q.data(), k, eligible, options.probes, options.stats);
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
    if ((codec_ && codec_->trained()) || ivf_trained()) {  // codes or lists: each query scans on its own (the tiles below are for full precision without lists)
        std::atomic<bool> stopped{false};
        auto one = [&](std::size_t q) {
            const detail::SearchStopScope stop(options, nullptr);
            PreparedVector pq(indexed_dim());
            prepare_truncated(pq, queries.subspan(q * dim(), dim()), dim(), metric_);
            results[q] = scan(pq.data(), k, eligible, options.probes, nullptr);
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
    std::vector<float> table;  // codes only: distances from codes
    if (codes_only()) codec_->prepare(std::span<const float>(q.data(), indexed_dim()), table);
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
            const float d = codes_only() ? codec_->distance(table, codes_.data() + std::size_t(node) * codec_->code_size())
                                         : index_distance(distance_(q.data(), vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine);
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