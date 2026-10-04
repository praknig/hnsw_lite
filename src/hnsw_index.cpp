/**
 * @file hnsw_index.cpp
 * @brief Implementation of HnswIndex (see hnsw_index.h for the design).
 *
 * Follows Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
 * search using Hierarchical Navigable Small World graphs" (arXiv:1603.09320),
 * with the same choices as hnswlib: a new node selects M neighbors on every
 * level, and lists may grow to 2*M on level 0 through reverse links.
 */
#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>

namespace vecdb {
namespace {

/// Throws if the settings cannot build a valid graph; otherwise returns them.
const HnswParams& validated(const HnswParams& p) {
    if (p.M < 2) throw std::invalid_argument("HnswParams::M must be at least 2");
    if (p.ef_construction < 1) throw std::invalid_argument("HnswParams::ef_construction must be >= 1");
    return p;
}

/// Highest level GraphStorage can store.
constexpr int kMaxLevel = 255;

}  // namespace

HnswIndex::HnswIndex(std::size_t dim, Metric metric, HnswParams params)
    : metric_(metric),
      distance_(get_distance(metric)),
      params_(validated(params)),
      level_factor_(1.0 / std::log(static_cast<double>(params.M))),
      storage_(dim, params.M),
      stride_(storage_.vectors().stride()),
      scratch_(dim),
      rng_(params.seed) {}

int HnswIndex::random_level() {
    // Uniform in [0, 1) from the top 53 bits. Done by hand instead of with
    // std::uniform_real_distribution so levels are identical on every compiler.
    const double u = static_cast<double>(rng_() >> 11) * 0x1.0p-53;
    const double level = -std::log(1.0 - u) * level_factor_;  // 1 - u is in (0, 1]
    return static_cast<int>(std::min(level, static_cast<double>(kMaxLevel)));
}

float HnswIndex::distance_to(const float* query, NodeId node) const {
    return distance_(query, storage_.vectors().get_padded(node).data(), stride_);
}

float HnswIndex::distance_between(NodeId a, NodeId b) const {
    return distance_to(storage_.vectors().get_padded(a).data(), b);
}

bool HnswIndex::same_vector(NodeId a, NodeId b) const {
    // For different vectors memcmp stops at the first differing byte, so this is cheap.
    const VectorStore& vs = storage_.vectors();
    return std::memcmp(vs.get(a).data(), vs.get(b).data(), vs.dim() * sizeof(float)) == 0;
}

Candidate HnswIndex::greedy_closest(const float* query, Candidate start, int level) const {
    Candidate best = start;
    for (bool moved = true; moved;) {
        moved = false;
        for (NodeId neighbor : storage_.graph().links(best.id, level)) {
            if (neighbor == kEmpty) break;
            const float d = distance_to(query, neighbor);
            if (d < best.distance) {
                best = {d, neighbor};
                moved = true;
            }
        }
    }
    return best;
}

std::vector<Candidate> HnswIndex::search_level(const float* query,
                                               const std::vector<Candidate>& entries,
                                               std::size_t ef, int level, VisitedList& visited,
                                               bool skip_removed) const {
    visited.reset(storage_.size());
    const IdMap& ids = storage_.ids();

    // candidates: nodes to expand, closest on top (min-heap).
    // results:    best `ef` nodes found, worst on top (TopK).
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidates;
    TopK results(ef);

    for (const Candidate& e : entries) {
        if (!visited.visit(e.id)) continue;
        candidates.push(e);
        if (!skip_removed || !ids.is_deleted(e.id)) results.push(e);
    }

    while (!candidates.empty()) {
        const Candidate current = candidates.top();
        // Nothing left can beat the current results: stop.
        if (results.full() && current.distance > results.worst_distance()) break;
        candidates.pop();

        for (NodeId neighbor : storage_.graph().links(current.id, level)) {
            if (neighbor == kEmpty) break;
            if (!visited.visit(neighbor)) continue;
            const float d = distance_to(query, neighbor);
            if (results.full() && d >= results.worst_distance()) continue;
            candidates.push({d, neighbor});  // removed nodes are still traversed...
            if (!skip_removed || !ids.is_deleted(neighbor))
                results.push({d, neighbor});  // ...but never returned
        }
    }
    return results.take_sorted();
}

std::vector<NodeId> HnswIndex::select_neighbors(const std::vector<Candidate>& sorted,
                                                std::size_t max) const {
    std::vector<NodeId> kept;
    kept.reserve(max);
    for (const Candidate& c : sorted) {
        if (kept.size() >= max) break;
        // Skip c if an already-kept neighbor is closer to it than the base node is
        // (c is then reachable through that neighbor), or is an exact copy of it.
        // Without the copy check, identical vectors (all at distance 0, a tie)
        // would fill every list near them and cut links to the rest of the graph.
        bool diverse = true;
        for (NodeId k : kept) {
            if (same_vector(c.id, k) || distance_between(c.id, k) < c.distance) {
                diverse = false;
                break;
            }
        }
        if (diverse) kept.push_back(c.id);
    }
    return kept;
}

void HnswIndex::connect(NodeId node, const std::vector<NodeId>& neighbors, int level) {
    GraphStorage& graph = storage_.graph();
    graph.set_links(node, level, neighbors);

    const std::size_t capacity = level == 0 ? graph.M0() : graph.M();
    for (NodeId neighbor : neighbors) {
        std::span<NodeId> slots = graph.links(neighbor, level);
        const std::size_t count = GraphStorage::count(slots);
        if (count < capacity) {  // room left: just add the reverse link
            slots[count] = node;
            continue;
        }
        // Full: choose the best `capacity` links among the old ones plus `node`.
        std::vector<Candidate> options;
        options.reserve(count + 1);
        options.push_back({distance_between(neighbor, node), node});
        for (std::size_t i = 0; i < count; ++i)
            options.push_back({distance_between(neighbor, slots[i]), slots[i]});
        std::sort(options.begin(), options.end());
        graph.set_links(neighbor, level, select_neighbors(options, capacity));
    }
}

void HnswIndex::add(std::uint64_t id, std::span<const float> vector) {
    // 1. Validate everything before changing anything (or drawing a random level).
    scratch_.prepare(vector, metric_);
    if (storage_.ids().find(id)) throw std::invalid_argument("external id already exists");

    // 2. Store the vector and its empty neighbor lists.
    const int level = random_level();
    const NodeId node = storage_.insert(id, scratch_.values(), level);
    ++live_;

    // 3. First node: it becomes the entry point.
    if (entry_ == kEmpty) {
        entry_ = node;
        max_level_ = level;
        return;
    }

    // 4. Greedy descent through the levels above the new node's level.
    const float* query = scratch_.data();
    Candidate current{distance_to(query, entry_), entry_};
    for (int l = max_level_; l > level; --l) current = greedy_closest(query, current, l);

    // 5. On each shared level, find candidates, pick neighbors, link both ways.
    auto visited = visited_pool_.acquire(storage_.size());
    std::vector<Candidate> entries{current};
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        // Removed nodes are traversed but not chosen as neighbors (as in hnswlib),
        // so new nodes do not waste link slots on them.
        std::vector<Candidate> found =
            search_level(query, entries, params_.ef_construction, l, *visited, true);
        if (found.empty())  // every nearby node is removed: link to them anyway,
            found = search_level(query, entries, params_.ef_construction, l, *visited, false);
                            // so the new node stays reachable from the entry point
        connect(node, select_neighbors(found, params_.M), l);
        entries = std::move(found);  // best nodes here start the next level down
    }

    // 6. A node higher than all others becomes the new entry point.
    if (level > max_level_) {
        max_level_ = level;
        entry_ = node;
    }
}

bool HnswIndex::remove(std::uint64_t id) {
    IdMap& ids = storage_.ids();
    auto node = ids.find(id);
    if (!node || ids.is_deleted(*node)) return false;
    ids.mark_deleted(*node);  // stays in the graph for navigation
    --live_;
    return true;
}

bool HnswIndex::contains(std::uint64_t id) const {
    auto node = storage_.ids().find(id);
    return node && !storage_.ids().is_deleted(*node);
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k,
                                            std::size_t ef) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);  // validates even when the index is empty
    if (k == 0 || live_ == 0) return {};
    k = std::min(k, live_);                          // cannot return more than exists
    ef = std::min(std::max(ef, k), storage_.size());  // at least k, at most every node

    // Greedy descent from the entry point to level 1.
    Candidate current{distance_to(q.data(), entry_), entry_};
    for (int l = max_level_; l > 0; --l) current = greedy_closest(q.data(), current, l);

    // Beam search on level 0, skipping removed nodes in the results.
    auto visited = visited_pool_.acquire(storage_.size());
    std::vector<Candidate> found = search_level(q.data(), {current}, ef, 0, *visited, true);
    if (found.size() > k) found.resize(k);

    std::vector<SearchResult> results;
    results.reserve(found.size());
    for (const Candidate& c : found) results.push_back({storage_.ids().external(c.id), c.distance});
    return results;
}

}  // namespace vecdb