/**
 * @file flat_index.cpp
 * @brief Implementation of FlatIndex (see flat_index.h for the design).
 */
#include "flat_index.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vecdb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric, Schema schema)
    : metric_(metric), distance_(get_distance(metric)), vectors_(dim), metadata_(std::move(schema)), scratch_(dim) {}

void FlatIndex::add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    const auto guard = lock_.write("add");
    add_unlocked(id, vector, metadata);
}

void FlatIndex::add_unlocked(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    // 1. Validate everything before changing anything.
    scratch_.prepare(vector, metric_);
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
        vectors_.add(scratch_.values());
    } catch (...) {
        ids_.undo_last_add(id);
        metadata_.undo_write(slot, undo);
        throw;
    }
    ++live_;
}

bool FlatIndex::remove(std::uint64_t id) {
    const auto guard = lock_.write("remove");
    const auto node = ids_.find(id);
    if (!node) return false;
    // Swap-with-last: move the last vector (and its metadata) into the freed slot,
    // then shrink by one. Every step below is non-allocating and cannot throw.
    const auto last = static_cast<NodeId>(vectors_.size() - 1);
    ids_.release(id);
    if (*node != last) {
        vectors_.move_row(last, *node);
        ids_.move_slot(last, *node);
        metadata_.move_row(last, *node);
    } else {
        metadata_.clear(last);
    }
    ids_.pop_back_slot();
    vectors_.pop_back();
    --live_;
    return true;
}

void FlatIndex::add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads) {
    (void)threads;  // Flat inserts are copies, done on the calling thread
    const auto guard = lock_.write("add_batch");
    add_batch_unlocked(ids, vectors, nullptr);
}

void FlatIndex::add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                          std::span<const Metadata> metadata, std::size_t threads) {
    (void)threads;
    if (metadata.size() != ids.size()) throw std::invalid_argument("add_batch: one metadata entry per vector is required");
    const auto guard = lock_.write("add_batch");
    add_batch_unlocked(ids, vectors, metadata.data());
}

void FlatIndex::add_batch_unlocked(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                                   const Metadata* metadata) {
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
            scratch_.prepare(vectors.subspan(i * d, d), metric_);
            if (ids_.find(ids[i]) || !seen.insert(ids[i]).second)
                throw std::invalid_argument("add_batch: an ID is already stored or appears twice in the batch");
        }
        if (metadata) metadata_.validate_batch(metadata, n);
    }

    // 2. Add in order; if memory runs out, take every added vector out again,
    //    newest first, which restores the index exactly.
    std::vector<MetadataStore::WriteUndo> undo;
    undo.reserve(n);
    try {
        for (std::size_t i = 0; i < n; ++i) {
            scratch_.prepare(vectors.subspan(i * d, d), metric_);
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
                vectors_.add(scratch_.values());
            } catch (...) {
                ids_.undo_last_add(ids[i]);
                metadata_.undo_write(slot, undo.back());
                undo.pop_back();
                throw;
            }
            ++live_;
        }
    } catch (...) {
        for (std::size_t k = undo.size(); k-- > 0;) {
            const auto slot = static_cast<NodeId>(vectors_.size() - 1);
            vectors_.pop_back();
            ids_.undo_last_add(ids[k]);
            metadata_.undo_write(slot, undo[k]);
            --live_;
        }
        throw;
    }
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
    scratch_.prepare(vector, metric_);  // validates; may throw before any change
    if (metadata) metadata_.validate(*metadata);
    const auto node = ids_.find(id);
    if (!node) return false;
    if (metadata) metadata_.write(*node, *metadata);  // all-or-nothing
    vectors_.overwrite(*node, scratch_.values());     // cannot fail: already validated
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
        TopK top(k);
        const std::size_t count = vectors_.size(), stride = vectors_.stride();
        std::size_t matches = 0;
        for (NodeId node = 0; node < count; ++node) {
            if (eligible.filtered() && !eligible(node)) continue;
            ++matches;
            top.push({index_distance(distance_(query, vectors_.get_padded(node).data(), stride), metric_ == Metric::Cosine), node});
        }
        st.nodes_visited = count;
        st.distance_computations = matches;
        st.estimated_matches = matches;
        st.exact_count = true;
        st.selectivity = double(matches) / double(live_);
        results.reserve(k);
        for (const Candidate& c : top.take_sorted()) results.push_back({ids_.external(c.id), c.distance});
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
    PreparedVector q(dim());
    q.prepare(query, metric_);  // validates even when the index is empty
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

    // Tiling: a block of stored vectors is compared with a chunk of queries, so
    // each vector loaded into the cache serves every query in the chunk. Rows are
    // pushed in increasing order for every query, exactly as in search().
    constexpr std::size_t kQueryChunk = 16, kRowBlock = 256;
    const std::size_t chunks = (nq + kQueryChunk - 1) / kQueryChunk;
    const std::size_t count = vectors_.size(), stride = vectors_.stride();
    auto run_chunk = [&](std::size_t chunk) {
        const IndexLock::Hold inside(lock_);  // pool threads work under the caller's read
        const std::size_t q0 = chunk * kQueryChunk, q1 = std::min(nq, q0 + kQueryChunk);
        std::vector<PreparedVector> prepared;
        prepared.reserve(q1 - q0);
        std::vector<TopK> tops;
        tops.reserve(q1 - q0);
        for (std::size_t q = q0; q < q1; ++q) {
            prepared.emplace_back(dim());
            prepared.back().prepare(queries.subspan(q * dim(), dim()), metric_);
            tops.emplace_back(k);
        }
        std::vector<std::uint8_t> ok(kRowBlock);
        for (std::size_t r0 = 0; r0 < count; r0 += kRowBlock) {
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
    return results;
}

std::vector<SearchResult> FlatIndex::search_range(std::span<const float> query, float radius,
                                                  std::size_t max_results, const SearchOptions& options) const {
    const auto guard = lock_.read();
    PreparedVector q(dim());
    q.prepare(query, metric_);
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