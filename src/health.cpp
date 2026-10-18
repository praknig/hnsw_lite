/**
 * @file health.cpp
 * @brief HnswIndex::health; see health.h.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "disk_index.h"
#include "flat_index.h"
#include "health.h"
#include "hnsw_index.h"
#include "kmeans.h"
#include "prepared_vector.h"

namespace vecdb {

namespace {

/// `nodes`, with its first `take` entries a random sample (a partial shuffle).
void sample_first(std::vector<NodeId>& nodes, std::size_t take, std::uint64_t seed) {
    SplitMix64 rng(seed);
    take = std::min(take, nodes.size());
    for (std::size_t i = 0; i < take; ++i) std::swap(nodes[i], nodes[i + std::size_t(rng() % (nodes.size() - i))]);
}

/// Squared reconstruction error relative to squared length, over `vectors`.
double relative_error(const VectorCodec& codec, const std::vector<std::span<const float>>& vectors, std::size_t dim) {
    std::vector<std::uint8_t> code(codec.code_size());
    std::vector<float> back(dim);
    double err = 0, len = 0;
    for (const auto v : vectors) {
        codec.encode(v, code.data());
        codec.decode(code.data(), back);
        for (std::size_t j = 0; j < dim; ++j) {
            err += (double(v[j]) - back[j]) * (double(v[j]) - back[j]);
            len += double(v[j]) * v[j];
        }
    }
    return len > 0 ? err / len : 0.0;
}

/// Compression figures: error on `eval`, SQ8 clipping, and the error of a codec
/// retrained on `train` (disjoint from `eval`: a codec flatters its training data).
void measure_codec(HealthReport& r, const VectorCodec& codec, const std::vector<std::span<const float>>& eval,
                   const std::vector<std::span<const float>>& train, std::size_t dim, Metric metric) {
    if (eval.empty()) return;
    r.codec_error = relative_error(codec, eval, dim);
    if (const auto* sq8 = dynamic_cast<const SQ8Codec*>(&codec)) {
        std::size_t outside = 0;
        for (const auto v : eval)
            for (std::size_t j = 0; j < dim; ++j) {
                const float lo = sq8->minimum()[j], hi = lo + 255.0f * sq8->step()[j];
                outside += v[j] < lo || v[j] > hi;
            }
        r.clipped = double(outside) / double(eval.size() * dim);
    }
    if (train.empty()) return;
    const auto* pq = dynamic_cast<const PQCodec*>(&codec);
    auto fresh = make_codec(codec.kind(), dim, metric, pq ? pq->parts() : 0);
    std::vector<float> sample;
    sample.reserve(train.size() * dim);
    for (const auto v : train) sample.insert(sample.end(), v.begin(), v.end());
    fresh->train(sample);
    r.codec_error_retrained = relative_error(*fresh, eval, dim);
}

double squared(std::span<const float> a, const float* b) {
    double d = 0;
    for (std::size_t j = 0; j < a.size(); ++j) d += (double(a[j]) - b[j]) * (double(a[j]) - b[j]);
    return d;
}

}  // namespace

HealthReport HnswIndex::health(const HealthOptions& options) const {
    HealthReport r;
    struct Probe {
        std::uint64_t id;
        std::vector<std::uint64_t> truth;  // exact top k, by ID
    };
    std::vector<Probe> probes;
    {
        const auto guard = lock_.read();
        const GraphStorage& g = storage_.graph();
        const IdMap& ids = storage_.ids();
        const std::size_t n = storage_.size();
        r.live = live_;
        r.slots = n;
        r.free_slots = free_slots_.size();
        r.removals = removals_since_upkeep_;
        r.updates = updates_since_upkeep_;
        if (live_ == 0) return r;

        // Levels and degrees, incoming links, stale links and links to removed nodes.
        r.levels.assign(std::size_t(max_level_ + 1), LevelHealth{});
        std::vector<bool> incoming(n, false);
        std::size_t degrees = 0;
        r.min_degree = SIZE_MAX;
        for (NodeId node = 0; node < n; ++node) {
            if (ids.is_deleted(node)) continue;
            for (int level = 0; level <= g.level(node); ++level) {
                ++r.levels[std::size_t(level)].nodes;
                const auto slots = g.links(node, level);
                const std::size_t count = GraphStorage::count(slots);
                for (std::size_t i = 0; i < count; ++i) {
                    const NodeId nb = slots[i];
                    if (g.level(nb) < level) ++r.stale_links;
                    if (level != 0) continue;
                    if (ids.is_deleted(nb)) ++r.links_to_removed;
                    else incoming[nb] = true;
                }
                if (level != 0) continue;
                degrees += count;
                r.min_degree = std::min(r.min_degree, count);
                if (count < g.M()) ++r.underfull;
            }
        }
        r.mean_degree = double(degrees) / double(live_);
        for (std::size_t l = 0; l < r.levels.size(); ++l)
            r.levels[l].expected = double(live_) * std::pow(double(params_.M), -double(l));
        for (NodeId node = 0; node < n; ++node)
            if (!ids.is_deleted(node) && !incoming[node] && node != entry_) ++r.no_incoming;

        // Reachability on level 0, as a search walks: through removed nodes too.
        std::vector<bool> reached(n, false);
        std::vector<NodeId> stack{entry_};
        reached[entry_] = true;
        std::size_t live_reached = 0;
        while (!stack.empty()) {
            const NodeId node = stack.back();
            stack.pop_back();
            if (!ids.is_deleted(node)) ++live_reached;
            for (NodeId nb : g.links(node, 0)) {
                if (nb == kEmpty) break;
                if (!reached[nb]) {
                    reached[nb] = true;
                    stack.push_back(nb);
                }
            }
        }
        r.unreachable = live_ - live_reached;

        // Exact answers for a sample of stored vectors (searched for below,
        // without the lock: search takes it itself). Needs the stored vectors.
        if (options.recall_sample > 0 && options.k > 0 && storage_.vectors().has_values()) {
            std::vector<NodeId> live_nodes;
            live_nodes.reserve(live_);
            for (NodeId node = 0; node < n; ++node)
                if (!ids.is_deleted(node)) live_nodes.push_back(node);
            SplitMix64 rng(options.seed);
            const std::size_t take = std::min(options.recall_sample, live_nodes.size());
            for (std::size_t i = 0; i < take; ++i)  // a partial shuffle: distinct samples
                std::swap(live_nodes[i], live_nodes[i + std::size_t(rng() % (live_nodes.size() - i))]);
            const bool cosine = metric_ == Metric::Cosine;
            std::vector<std::pair<float, NodeId>> all(live_nodes.size());
            for (std::size_t i = 0; i < take; ++i) {
                const float* q = storage_.vectors().get_padded(live_nodes[i]).data();
                for (std::size_t j = 0; j < live_nodes.size(); ++j)
                    all[j] = {index_distance(distance_(q, storage_.vectors().get_padded(live_nodes[j]).data(), stride_), cosine),
                              live_nodes[j]};
                const std::size_t k = std::min(options.k, all.size());
                std::partial_sort(all.begin(), all.begin() + std::ptrdiff_t(k), all.end());
                Probe p{ids.external(live_nodes[i]), {}};
                for (std::size_t j = 0; j < k; ++j) p.truth.push_back(ids.external(all[j].second));
                probes.push_back(std::move(p));
            }
        }
        if (codec_ && codec_->trained() && storage_.vectors().has_values() && options.codec_sample > 0) {
            std::vector<NodeId> nodes;
            for (NodeId node = 0; node < n; ++node)
                if (!ids.is_deleted(node)) nodes.push_back(node);
            sample_first(nodes, options.codec_sample + options.retrain_sample, options.seed + 1);
            std::vector<std::span<const float>> eval, train;
            for (std::size_t i = 0; i < nodes.size() && i < options.codec_sample + options.retrain_sample; ++i)
                (i < options.codec_sample ? eval : train).push_back(storage_.vectors().get(nodes[i]));
            measure_codec(r, *codec_, eval, train, storage_.vectors().dim(), metric_);
        }
    }
    // Searches for the sampled vectors: a snapshot, so changes made meanwhile can
    // move the estimate slightly.
    if (!probes.empty()) {
        std::size_t hits = 0, wanted = 0;
        for (const Probe& p : probes) {
            const auto found = search_by_id(p.id, p.truth.size(), std::max(options.ef, p.truth.size()), {}, true);
            if (!found) continue;  // GCOVR_EXCL_BR_LINE: removed meanwhile (a race between the two steps; timing only)
            const std::unordered_set<std::uint64_t> truth(p.truth.begin(), p.truth.end());
            for (const auto& x : *found) hits += truth.count(x.id);
            wanted += p.truth.size();
        }
        if (wanted > 0) r.recall = double(hits) / double(wanted);
    }
    return r;
}

HealthReport FlatIndex::health(const HealthOptions& options) const {
    HealthReport r;
    struct Probe {
        std::uint64_t id;
        std::vector<std::uint64_t> truth;
    };
    std::vector<Probe> probes;
    {
        const auto guard = lock_.read();
        r.live = live_;
        r.slots = live_;  // a flat index keeps its rows packed: no free slots
        r.removals = removals_since_upkeep_;
        r.updates = updates_since_upkeep_;
        r.lists = ivf_.lists;
        r.lists_trained = !lists_.empty();
        if (r.lists_trained) {
            for (const auto& l : lists_) {
                r.largest_list = std::max(r.largest_list, l.size());
                r.empty_lists += l.empty();
            }
            if (live_ > 0) r.list_imbalance = double(r.largest_list) / (double(live_) / double(lists_.size()));
        }
        if (live_ == 0 || !vectors_.has_values()) return r;  // codes only: no exact answers, no errors
        const std::size_t dim = vectors_.dim(), stride = vectors_.stride();
        std::vector<NodeId> rows(live_);
        for (NodeId i = 0; i < live_; ++i) rows[i] = i;
        // Exact answers for a sample (searched for below, without the lock).
        if (options.recall_sample > 0 && options.k > 0) {
            sample_first(rows, options.recall_sample, options.seed);
            const bool cosine = metric_ == Metric::Cosine;
            std::vector<std::pair<float, NodeId>> all(live_);
            const std::size_t take = std::min(options.recall_sample, std::size_t(live_));
            for (std::size_t i = 0; i < take; ++i) {
                const float* q = vectors_.get_padded(rows[i]).data();
                for (NodeId j = 0; j < live_; ++j) all[j] = {index_distance(distance_(q, vectors_.get_padded(j).data(), stride), cosine), j};
                const std::size_t k = std::min(options.k, all.size());
                std::partial_sort(all.begin(), all.begin() + std::ptrdiff_t(k), all.end());
                Probe p{ids_.external(rows[i]), {}};
                for (std::size_t j = 0; j < k; ++j) p.truth.push_back(ids_.external(all[j].second));
                probes.push_back(std::move(p));
            }
        }
        // Compression and IVF figures on a second sample (and retraining on a third).
        if (options.codec_sample > 0 && ((codec_ && codec_->trained()) || r.lists_trained)) {
            for (NodeId i = 0; i < live_; ++i) rows[i] = i;
            sample_first(rows, options.codec_sample + options.retrain_sample, options.seed + 1);
            std::vector<std::span<const float>> eval, train;
            for (std::size_t i = 0; i < rows.size() && i < options.codec_sample + options.retrain_sample; ++i)
                (i < options.codec_sample ? eval : train).push_back(vectors_.get(rows[i]));
            if (codec_ && codec_->trained()) measure_codec(r, *codec_, eval, train, dim, metric_);
            if (r.lists_trained && !eval.empty()) {
                double total = 0;
                for (std::size_t i = 0; i < eval.size(); ++i) total += squared(eval[i], &centroids_[std::size_t(list_of_[rows[i]]) * stride]);
                r.centroid_distance = total / double(eval.size());
                if (train.size() >= lists_.size()) {
                    std::vector<float> sample;
                    for (const auto v : train) sample.insert(sample.end(), v.begin(), v.end());
                    const KMeansResult fresh = kmeans(sample, dim, KMeansParams{lists_.size(), 25, options.seed, 1});
                    const std::size_t k = fresh.centroids.size() / dim;
                    double best_total = 0;
                    for (const auto v : eval) {
                        double best = std::numeric_limits<double>::infinity();
                        for (std::size_t c = 0; c < k; ++c) best = std::min(best, squared(v, &fresh.centroids[c * dim]));
                        best_total += best;
                    }
                    r.centroid_distance_retrained = best_total / double(eval.size());
                }
            }
        }
    }
    if (!probes.empty()) {
        std::size_t hits = 0, wanted = 0;
        for (const Probe& p : probes) {
            const auto found = search_by_id(p.id, p.truth.size(), {}, true);
            if (!found) continue;  // GCOVR_EXCL_BR_LINE: removed meanwhile (a race between the two steps; timing only)
            const std::unordered_set<std::uint64_t> truth(p.truth.begin(), p.truth.end());
            for (const auto& x : *found) hits += truth.count(x.id);
            wanted += p.truth.size();
        }
        if (wanted > 0) r.recall = double(hits) / double(wanted);
    }
    return r;
}

HealthReport DiskIndex::health(const HealthOptions& options) const {
    HealthReport r;
    const std::size_t n = size();
    r.live = r.slots = n;
    r.levels = {LevelHealth{n, double(n)}};
    // Samples: for recall (exact answers found during the pass), and for PQ.
    std::vector<NodeId> nodes(n);
    for (NodeId i = 0; i < n; ++i) nodes[i] = i;
    sample_first(nodes, options.recall_sample, options.seed);
    const std::size_t probes = options.k > 0 ? std::min(options.recall_sample, n) : 0;
    std::vector<NodeId> probe_nodes(nodes.begin(), nodes.begin() + std::ptrdiff_t(probes));
    std::vector<PreparedVector> queries;
    for (NodeId q : probe_nodes) {
        queries.emplace_back(dim_);
        queries.back().prepare(read_node(q).vector, Metric::L2);  // already prepared: copied and padded
    }
    std::vector<std::vector<std::pair<float, NodeId>>> best(probes);  // max-heaps of the k nearest so far
    std::vector<NodeId> pq_nodes(n);
    for (NodeId i = 0; i < n; ++i) pq_nodes[i] = i;
    sample_first(pq_nodes, options.codec_sample + options.retrain_sample, options.seed + 1);
    std::vector<int> pq_role(n, 0);  // 1: evaluation, 2: retraining
    for (std::size_t i = 0; i < pq_nodes.size() && i < options.codec_sample + options.retrain_sample; ++i)
        pq_role[pq_nodes[i]] = i < options.codec_sample ? 1 : 2;
    std::vector<std::vector<float>> eval_vectors, train_vectors;

    // One pass in node order: neighbor lists, distances to the probes, PQ samples.
    std::vector<std::vector<NodeId>> adjacency(n);
    PreparedVector row(dim_);
    const bool cosine = metric_ == Metric::Cosine;
    const std::size_t k = std::min(options.k, n);
    std::size_t degrees = 0;
    r.min_degree = SIZE_MAX;
    for (NodeId i = 0; i < n; ++i) {
        Node node = read_node(i);
        if (!probe_nodes.empty()) {
            row.prepare(node.vector, Metric::L2);
            for (std::size_t q = 0; q < probes; ++q) {
                const float d = index_distance(distance_(queries[q].data(), row.data(), row.stride()), cosine);
                auto& h = best[q];
                if (h.size() < k) {
                    h.emplace_back(d, i);
                    std::push_heap(h.begin(), h.end());
                } else if (std::make_pair(d, i) < h.front()) {
                    std::pop_heap(h.begin(), h.end());
                    h.back() = {d, i};
                    std::push_heap(h.begin(), h.end());
                }
            }
        }
        if (pq_role[i] == 1) eval_vectors.push_back(node.vector);
        if (pq_role[i] == 2) train_vectors.push_back(node.vector);
        degrees += node.neighbors.size();
        r.min_degree = std::min(r.min_degree, node.neighbors.size());
        if (2 * node.neighbors.size() < R_) ++r.underfull;
        adjacency[i] = std::move(node.neighbors);
    }
    r.mean_degree = double(degrees) / double(n);
    std::vector<bool> incoming(n, false), reached(n, false);
    for (const auto& list : adjacency)
        for (NodeId nb : list) incoming[nb] = true;
    for (NodeId i = 0; i < n; ++i) r.no_incoming += !incoming[i] && i != medoid_;
    std::vector<NodeId> stack{medoid_};
    reached[medoid_] = true;
    std::size_t count = 0;
    while (!stack.empty()) {
        const NodeId node = stack.back();
        stack.pop_back();
        ++count;
        for (NodeId nb : adjacency[node])
            if (!reached[nb]) {
                reached[nb] = true;
                stack.push_back(nb);
            }
    }
    r.unreachable = n - count;
    // Compression: the PQ codes guide every search.
    if (!eval_vectors.empty()) {
        std::vector<std::span<const float>> eval(eval_vectors.begin(), eval_vectors.end()), train(train_vectors.begin(), train_vectors.end());
        measure_codec(r, *codec_, eval, train, dim_, metric_);
    }
    // Searches for the probes.
    if (probes > 0) {
        std::size_t hits = 0, wanted = 0;
        for (std::size_t q = 0; q < probes; ++q) {
            std::unordered_set<std::uint64_t> truth;
            for (const auto& [d, node] : best[q]) truth.insert(ids_[node]);
            const auto found = search_by_id(ids_[probe_nodes[q]], k, DiskSearchParams{std::max(options.ef, k), 4}, {}, true);
            for (const auto& x : *found) hits += truth.count(x.id);
            wanted += truth.size();
        }
        r.recall = double(hits) / double(wanted);
    }
    return r;
}

}  // namespace vecdb