/**
 * @file vamana.cpp
 * @brief Building and searching a Vamana graph; see vamana.h.
 */
#include "vamana.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "prepared_vector.h"

namespace vecdb {

namespace {

/// SplitMix64: portable, so the same seed gives the same graph everywhere.
struct Random {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

}  // namespace

void validate(const VamanaParams& p) {
    if (p.R < 1) throw std::invalid_argument("VamanaParams::R must be at least 1");
    if (p.L < p.R) throw std::invalid_argument("VamanaParams::L must be at least R");
    if (!(p.alpha >= 1.0f) || !std::isfinite(p.alpha)) throw std::invalid_argument("VamanaParams::alpha must be at least 1");
}

VamanaGraph::VamanaGraph(std::size_t dim, Metric metric, const VamanaParams& params)
    : vectors_(dim), metric_(metric), distance_(get_distance(metric)), params_(params) {}

float VamanaGraph::distance(const float* a, NodeId b) const {
    return index_distance(distance_(a, vectors_.get_padded(b).data(), vectors_.stride()), metric_ == Metric::Cosine);
}

VamanaGraph VamanaGraph::build(std::span<const float> vectors, std::size_t dim, Metric metric, const VamanaParams& params) {
    validate(params);
    if (metric != Metric::L2 && metric != Metric::Cosine) throw std::invalid_argument("Vamana: L2 and cosine only");
    if (dim == 0 || vectors.empty() || vectors.size() % dim != 0) throw std::invalid_argument("Vamana: vectors must be whole, at least one");
    const std::size_t n = vectors.size() / dim;
    if (n >= std::size_t(kEmpty)) throw std::invalid_argument("Vamana: too many vectors");  // GCOVR_EXCL_LINE: needs 4 billion vectors
    VamanaGraph g(dim, metric, params);
    PreparedVector prepared(dim);
    for (std::size_t i = 0; i < n; ++i) {
        prepared.prepare(vectors.subspan(i * dim, dim), metric);  // checks the values; normalizes for cosine
        g.vectors_.add(prepared.values());
    }
    g.adjacency_.assign(n, {});
    g.mark_.assign(n, 0);
    const std::size_t R = std::min(params.R, n - 1);
    Random rng{params.seed};

    // The medoid: the vector nearest to the mean.
    std::vector<double> mean(dim, 0.0);
    for (NodeId i = 0; i < n; ++i)
        for (std::size_t d = 0; d < dim; ++d) mean[d] += g.vectors_.get(i)[d];
    std::vector<float> centre(g.vectors_.stride(), 0.0f);
    for (std::size_t d = 0; d < dim; ++d) centre[d] = static_cast<float>(mean[d] / double(n));
    float best = 0;
    for (NodeId i = 0; i < n; ++i) {
        const float d = index_distance(get_distance(Metric::L2)(centre.data(), g.vectors_.get_padded(i).data(), g.vectors_.stride()), false);
        if (i == 0 || d < best) {
            best = d;
            g.medoid_ = i;
        }
    }

    // A random graph of degree R to start from.
    for (NodeId p = 0; p < n; ++p) {
        auto& out = g.adjacency_[p];
        if (n - 1 <= R) {  // few vectors: every other one
            for (NodeId q = 0; q < n; ++q)
                if (q != p) out.push_back(q);
            continue;
        }
        while (out.size() < R) {
            const NodeId q = static_cast<NodeId>(rng.next() % n);
            if (q != p && std::find(out.begin(), out.end(), q) == out.end()) out.push_back(q);
        }
    }
    if (n - 1 <= R) return g;  // few vectors: the complete graph is already the best possible

    // Two passes: alpha = 1, then the given alpha; nodes in a random order.
    std::vector<NodeId> order(n);
    for (NodeId i = 0; i < n; ++i) order[i] = i;
    std::vector<std::pair<float, NodeId>> beam;
    for (const float alpha : {1.0f, params.alpha}) {
        for (std::size_t i = n - 1; i > 0; --i) std::swap(order[i], order[rng.next() % (i + 1)]);
        for (NodeId p : order) {
            const float* x = g.vectors_.get_padded(p).data();
            auto candidates = g.greedy(x, params.L, beam);
            for (NodeId q : g.adjacency_[p]) candidates.emplace_back(g.distance(x, q), q);
            g.adjacency_[p] = g.robust_prune(p, std::move(candidates), alpha);
            for (NodeId j : g.adjacency_[p]) {  // reverse edges
                auto& back = g.adjacency_[j];
                if (std::find(back.begin(), back.end(), p) != back.end()) continue;
                if (back.size() < R) {
                    back.push_back(p);
                } else {
                    const float* y = g.vectors_.get_padded(j).data();
                    std::vector<std::pair<float, NodeId>> c;
                    c.reserve(back.size() + 1);
                    for (NodeId q : back) c.emplace_back(g.distance(y, q), q);
                    c.emplace_back(g.distance(y, p), p);
                    back = g.robust_prune(j, std::move(c), alpha);
                }
            }
        }
    }
    g.repair_connectivity();
    return g;
}

std::vector<bool> VamanaGraph::reachable() const {
    std::vector<bool> seen(size(), false);
    std::vector<NodeId> queue{medoid_};
    seen[medoid_] = true;
    for (std::size_t head = 0; head < queue.size(); ++head)
        for (NodeId nb : adjacency_[queue[head]])
            if (!seen[nb]) {
                seen[nb] = true;
                queue.push_back(nb);
            }
    return seen;
}

void VamanaGraph::repair_connectivity() {
    const std::size_t R = std::min(params_.R, size() - 1);
    std::vector<std::pair<float, NodeId>> beam;
    for (int round = 0; round < 16; ++round) {
        const std::vector<bool> seen = reachable();
        std::vector<std::uint32_t> indegree(size(), 0);
        for (const auto& out : adjacency_)
            for (NodeId x : out) ++indegree[x];
        bool changed = false;
        for (NodeId u = 0; u < size(); ++u) {
            if (seen[u]) continue;
            // Reachable nodes near u, nearest first: a search for u visits only
            // reachable nodes. Prefer one with a free slot; otherwise replace a
            // neighbor that keeps another incoming edge (so it stays reachable);
            // only as a last resort the nearest node's farthest neighbor.
            auto visited = greedy(vectors_.get_padded(u).data(), params_.L, beam);
            std::erase_if(visited, [&](const auto& c) { return c.second == u || !seen[c.second]; });
            std::sort(visited.begin(), visited.end());
            NodeId host = medoid_;
            std::size_t slot = R;  // R: append
            bool placed = false;
            for (const auto& [d, v] : visited)
                if (adjacency_[v].size() < R) {
                    host = v;
                    placed = true;
                    break;
                }
            for (std::size_t i = 0; !placed && i < visited.size(); ++i) {
                const auto& out = adjacency_[visited[i].second];
                for (std::size_t s = out.size(); s-- > 0;)
                    if (indegree[out[s]] >= 2) {
                        host = visited[i].second;
                        slot = s;
                        placed = true;
                        break;
                    }
            }
            if (!placed) {
                if (!visited.empty()) host = visited.front().second;
                slot = adjacency_[host].size() - 1;  // the farthest neighbor
            }
            auto& out = adjacency_[host];
            if (std::find(out.begin(), out.end(), u) != out.end()) continue;
            if (slot == R || slot >= out.size()) {
                out.push_back(u);
            } else {
                --indegree[out[slot]];
                out[slot] = u;
            }
            ++indegree[u];
            changed = true;
        }
        if (!changed) return;
    }
}

std::vector<std::pair<float, NodeId>> VamanaGraph::greedy(const float* query, std::size_t beam_width,
                                                           std::vector<std::pair<float, NodeId>>& best) const {
    if (++generation_ == 0) {  // wrapped: start the marks again
        std::fill(mark_.begin(), mark_.end(), 0u);
        generation_ = 1;
    }
    std::vector<std::pair<float, NodeId>> expanded;
    std::vector<bool> done;  // per entry of `best`: expanded already
    best.clear();
    best.emplace_back(distance(query, medoid_), medoid_);
    done.push_back(false);
    mark_[medoid_] = generation_;
    for (;;) {
        std::size_t next = best.size();  // the closest candidate not expanded yet
        for (std::size_t i = 0; i < best.size(); ++i)
            if (!done[i]) {
                next = i;
                break;
            }
        if (next == best.size()) break;
        done[next] = true;
        const NodeId node = best[next].second;
        expanded.push_back(best[next]);
        for (NodeId nb : adjacency_[node]) {
            if (mark_[nb] == generation_) continue;
            mark_[nb] = generation_;
            const float d = distance(query, nb);
            if (best.size() >= beam_width && d >= best.back().first) continue;
            const auto at = std::upper_bound(best.begin(), best.end(), std::make_pair(d, nb));
            const std::size_t pos = std::size_t(at - best.begin());
            best.insert(at, {d, nb});
            done.insert(done.begin() + std::ptrdiff_t(pos), false);
            if (best.size() > beam_width) {
                best.pop_back();
                done.pop_back();
            }
        }
    }
    return expanded;
}

std::vector<NodeId> VamanaGraph::robust_prune(NodeId p, std::vector<std::pair<float, NodeId>> candidates, float alpha) const {
    // Equal distances (identical vectors) are ordered by a hash of (p, candidate),
    // not by the candidate's number: otherwise every node would prefer the same
    // lowest-numbered duplicates, which would become hubs while the others became
    // unreachable. Deterministic, and different for every node.
    auto tie = [p](NodeId q) {
        std::uint64_t z = (std::uint64_t(p) << 32 | q) + 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        if (a.second == b.second) return false;
        return tie(a.second) < tie(b.second);
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                 [](const auto& a, const auto& b) { return a.second == b.second; }),
                     candidates.end());
    // One node appears with one distance and one tie key, so its repeats are
    // adjacent after sorting: unique suffices.
    const std::size_t R = std::min(params_.R, size() - 1);
    std::vector<NodeId> kept;
    std::vector<bool> dropped(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size() && kept.size() < R; ++i) {
        if (dropped[i] || candidates[i].second == p) continue;
        const NodeId star = candidates[i].second;
        kept.push_back(star);
        const float* xs = vectors_.get_padded(star).data();
        // Drop the candidates `star` reaches alpha times more closely. A candidate
        // identical to p (distance 0) is never dropped: nothing reaches it better,
        // and dropping it would leave identical vectors with one neighbor each.
        for (std::size_t j = i + 1; j < candidates.size(); ++j)
            if (!dropped[j] && candidates[j].first > 0 && alpha * distance(xs, candidates[j].second) <= candidates[j].first)
                dropped[j] = true;
    }
    return kept;
}

std::vector<SearchResult> VamanaGraph::search(std::span<const float> query, std::size_t k, std::size_t L,
                                              std::size_t* expanded) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);
    if (k == 0) {
        if (expanded) *expanded = 0;
        return {};
    }
    std::vector<std::pair<float, NodeId>> best;
    const auto visited = greedy(q.data(), std::max(L, k), best);
    if (expanded) *expanded = visited.size();
    std::vector<SearchResult> results;
    for (std::size_t i = 0; i < best.size() && i < k; ++i) results.push_back({best[i].second, best[i].first});
    return results;
}

}  // namespace vecdb