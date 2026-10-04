/**
 * @file hnsw_index.cpp
 * @brief Implementation of HnswIndex (see hnsw_index.h for the design).
 *
 * Follows Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
 * search using Hierarchical Navigable Small World graphs" (arXiv:1603.09320),
 * with the same choices as hnswlib: a new node selects M neighbors on every
 * level, lists may grow to 2*M on level 0 through reverse links, and removal
 * repairs the removed node's neighbors.
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
    return ordered_distance(distance_(query, storage_.vectors().get_padded(node).data(), stride_));
}

float HnswIndex::distance_between(NodeId a, NodeId b) const {
    return distance_to(storage_.vectors().get_padded(a).data(), b);
}

bool HnswIndex::same_vector(NodeId a, NodeId b) const {
    // For different vectors memcmp stops at the first differing byte, so this is cheap.
    const VectorStore& vs = storage_.vectors();
    return std::memcmp(vs.get(a).data(), vs.get(b).data(), vs.dim() * sizeof(float)) == 0;
}

Candidate HnswIndex::greedy_closest(const float* query, Candidate start, int level,
                                    NodeId exclude) const {
    const GraphStorage& graph = storage_.graph();
    Candidate best = start;
    for (bool moved = true; moved;) {
        moved = false;
        for (NodeId neighbor : graph.links(best.id, level)) {
            if (neighbor == kEmpty) break;
            if (neighbor == exclude || graph.level(neighbor) < level) continue;  // stale link
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
                                               bool skip_removed, NodeId exclude) const {
    visited.reset(storage_.size());
    if (exclude != kEmpty) visited.visit(exclude);  // never step onto the excluded node
    const IdMap& ids = storage_.ids();
    const GraphStorage& graph = storage_.graph();

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

        for (NodeId neighbor : graph.links(current.id, level)) {
            if (neighbor == kEmpty) break;
            if (graph.level(neighbor) < level) continue;  // stale link to a reused slot
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
    const IdMap& ids = storage_.ids();
    graph.set_links(node, level, neighbors);

    const std::size_t capacity = level == 0 ? graph.M0() : graph.M();
    for (NodeId neighbor : neighbors) {
        std::span<NodeId> slots = graph.links(neighbor, level);
        const std::size_t count = GraphStorage::count(slots);
        const auto used_end = slots.begin() + static_cast<std::ptrdiff_t>(count);
        if (std::find(slots.begin(), used_end, node) != used_end) continue;  // already linked
        if (count < capacity) {  // room left: just add the reverse link
            slots[count] = node;
            continue;
        }
        // Full: choose the best `capacity` links among the old ones plus `node`,
        // dropping links to removed nodes and stale links on the way.
        std::vector<Candidate> options;
        options.reserve(count + 1);
        options.push_back({distance_between(neighbor, node), node});
        for (std::size_t i = 0; i < count; ++i) {
            const NodeId c = slots[i];
            if (ids.is_deleted(c) || graph.level(c) < level) continue;
            options.push_back({distance_between(neighbor, c), c});
        }
        std::sort(options.begin(), options.end());
        graph.set_links(neighbor, level, select_neighbors(options, capacity));
    }
}

void HnswIndex::add(std::uint64_t id, std::span<const float> vector) {
    // 1. Validate everything before changing anything (or drawing a random level).
    scratch_.prepare(vector, metric_);
    if (storage_.ids().find(id)) throw std::invalid_argument("external id already exists");
    reserve_one_more(free_slots_);  // so a failed insert can always hand its slot back

    // 2. Store the vector and its empty neighbor lists, reusing a free slot if any.
    const int level = random_level();
    NodeId node;
    if (!free_slots_.empty()) {
        node = free_slots_.back();
        storage_.insert_into(node, id, scratch_.values(), level);  // all-or-nothing
        free_slots_.pop_back();
    } else {
        node = storage_.insert(id, scratch_.values(), level);  // all-or-nothing
    }
    ++live_;

    // 3. No live node yet: this one becomes the entry point.
    if (entry_ == kEmpty) {
        entry_ = node;
        max_level_ = level;
        return;
    }

    try {
        link_new_node(node, level);
    } catch (...) {
        // Out of memory while linking: the node may be half-linked. Remove it
        // like any other vector: free its ID and put its slot back on the free list.
        storage_.ids().release(id);
        --live_;
        free_slots_.push_back(node);  // cannot throw: capacity reserved in step 1
        throw;
    }
}

void HnswIndex::link_new_node(NodeId node, int level) {
    // 4. Greedy descent through the levels above the new node's level.
    const float* query = scratch_.data();
    Candidate current{distance_to(query, entry_), entry_};
    for (int l = max_level_; l > level; --l) current = greedy_closest(query, current, l, node);

    // 5. On each shared level, find candidates, pick neighbors, link both ways.
    auto visited = visited_pool_.acquire(storage_.size());
    std::vector<Candidate> entries{current};
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        // Removed nodes are traversed but not chosen as neighbors (as in hnswlib),
        // so new nodes do not waste link slots on them.
        std::vector<Candidate> found =
            search_level(query, entries, params_.ef_construction, l, *visited, true, node);
        if (found.empty())  // every nearby node is removed: link to them anyway,
            found = search_level(query, entries, params_.ef_construction, l, *visited, false, node);
                            // so the new node stays reachable from the entry point
        connect(node, select_neighbors(found, params_.M), l);
        if (!found.empty()) entries = std::move(found);  // best nodes start the next level
    }

    // 6. A node higher than all others becomes the new entry point.
    if (level > max_level_) {
        max_level_ = level;
        entry_ = node;
    }
}

void HnswIndex::repair_neighbors(NodeId node) {
    GraphStorage& graph = storage_.graph();
    const IdMap& ids = storage_.ids();
    for (int l = 0; l <= graph.level(node); ++l) {
        // The removed node's live neighbors on this level.
        std::vector<NodeId> around;
        for (NodeId nb : graph.links(node, l)) {
            if (nb == kEmpty) break;
            if (nb != node && !ids.is_deleted(nb) && graph.level(nb) >= l) around.push_back(nb);
        }
        const std::size_t capacity = l == 0 ? graph.M0() : graph.M();
        for (NodeId n : around) {
            // Options: n's own links plus the removed node's neighbors, minus the
            // removed node, n itself, removed nodes and stale links.
            std::vector<Candidate> options;
            auto consider = [&](NodeId c) {
                if (c == n || c == node || ids.is_deleted(c) || graph.level(c) < l) return;
                options.push_back({distance_between(n, c), c});
            };
            for (NodeId c : graph.links(n, l)) {
                if (c == kEmpty) break;
                consider(c);
            }
            for (NodeId c : around) consider(c);
            std::sort(options.begin(), options.end());
            options.erase(std::unique(options.begin(), options.end(),
                                      [](const Candidate& a, const Candidate& b) { return a.id == b.id; }),
                          options.end());
            graph.set_links(n, l, select_neighbors(options, capacity));
        }
    }
}

void HnswIndex::choose_new_entry() {
    entry_ = kEmpty;
    max_level_ = -1;
    const IdMap& ids = storage_.ids();
    const GraphStorage& graph = storage_.graph();
    for (NodeId n = 0; n < storage_.size(); ++n) {
        if (!ids.is_deleted(n) && graph.level(n) > max_level_) {
            max_level_ = graph.level(n);
            entry_ = n;
        }
    }
}

bool HnswIndex::remove(std::uint64_t id) {
    IdMap& ids = storage_.ids();
    const auto node = ids.find(id);
    if (!node) return false;
    reserve_one_more(free_slots_);  // may throw; nothing changed yet

    // Repair first, while the node is still live: if memory runs out here, the
    // vector simply stays stored and every list touched so far is still valid.
    if (params_.repair_on_remove) repair_neighbors(*node);

    // From here on nothing can throw.
    ids.release(id);
    --live_;
    free_slots_.push_back(*node);
    if (*node == entry_) choose_new_entry();
    return true;
}

bool HnswIndex::contains(std::uint64_t id) const { return storage_.ids().find(id).has_value(); }

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k,
                                            std::size_t ef) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);  // validates even when the index is empty
    if (k == 0 || live_ == 0) return {};
    k = std::min(k, live_);                          // cannot return more than exists
    ef = std::min(std::max(ef, k), storage_.size());  // at least k, at most every node

    // Greedy descent from the entry point to level 1.
    Candidate current{distance_to(q.data(), entry_), entry_};
    for (int l = max_level_; l > 0; --l) current = greedy_closest(q.data(), current, l, kEmpty);

    // Beam search on level 0, skipping removed nodes in the results.
    auto visited = visited_pool_.acquire(storage_.size());
    std::vector<Candidate> found = search_level(q.data(), {current}, ef, 0, *visited, true, kEmpty);
    if (found.size() > k) found.resize(k);

    std::vector<SearchResult> results;
    results.reserve(found.size());
    for (const Candidate& c : found) results.push_back({storage_.ids().external(c.id), c.distance});
    return results;
}

CompactStats HnswIndex::compact() {
    const std::size_t reclaimed = deleted_count();
    // Build the replacement completely first, so a failure leaves this index intact.
    HnswIndex fresh(dim(), metric_, params_);
    const IdMap& ids = storage_.ids();
    for (NodeId n = 0; n < storage_.size(); ++n)
        if (!ids.is_deleted(n)) fresh.add(ids.external(n), storage_.vectors().get(n));

    // Commit: moves only, nothing can throw.
    storage_ = std::move(fresh.storage_);
    rng_ = fresh.rng_;
    entry_ = fresh.entry_;
    max_level_ = fresh.max_level_;
    live_ = fresh.live_;
    free_slots_ = std::move(fresh.free_slots_);
    return {live_, reclaimed};
}

}  // namespace vecdb