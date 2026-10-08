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

HnswIndex::HnswIndex(std::size_t dim, Metric metric, HnswParams params, Schema schema)
    : metric_(metric),
      distance_(get_distance(metric)),
      params_(validated(params)),
      level_factor_(1.0 / std::log(static_cast<double>(params.M))),
      storage_(dim, params.M),
      stride_(storage_.vectors().stride()),
      scratch_(dim),
      rng_(params.seed),
      metadata_(std::move(schema)) {}

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
                                               NodeId exclude, const Eligibility& eligible,
                                               SearchStats* stats) const {
    visited.reset(storage_.size());
    if (exclude != kEmpty) visited.visit(exclude);  // never step onto the excluded node
    const GraphStorage& graph = storage_.graph();

    // candidates: nodes to expand, closest on top (min-heap).
    // results:    best `ef` eligible nodes found, worst on top (TopK).
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidates;
    TopK results(ef);

    for (const Candidate& e : entries) {
        if (!visited.visit(e.id)) continue;  // GCOVR_EXCL_BR_LINE: defensive, entries are always unique
        candidates.push(e);
        if (eligible(e.id)) results.push(e);
    }

    std::size_t expanded = 0, computed = 0;
    while (!candidates.empty()) {
        const Candidate current = candidates.top();
        // Nothing left can beat the current results: stop.
        if (results.full() && current.distance > results.worst_distance()) break;
        candidates.pop();
        ++expanded;

        for (NodeId neighbor : graph.links(current.id, level)) {
            if (neighbor == kEmpty) break;
            if (graph.level(neighbor) < level) continue;  // stale link to a reused slot
            if (!visited.visit(neighbor)) continue;
            const float d = distance_to(query, neighbor);
            ++computed;
            if (results.full() && d >= results.worst_distance()) continue;
            candidates.push({d, neighbor});                       // ineligible nodes are traversed...
            if (eligible(neighbor)) results.push({d, neighbor});  // ...but never returned
        }
    }
    if (stats) {
        stats->nodes_visited += expanded;
        stats->distance_computations += computed;
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

void HnswIndex::add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    // 1. Validate everything before changing anything (or drawing a random level).
    scratch_.prepare(vector, metric_);
    metadata_.validate(metadata);
    if (storage_.ids().find(id)) throw std::invalid_argument("external id already exists");
    reserve_one_more(free_slots_);  // so a failed insert can always hand its slot back

    // 2. Metadata first (into the slot the vector will take), then the vector and
    //    its empty neighbor lists, reusing a free slot if any. All-or-nothing.
    const NodeId slot = free_slots_.empty() ? static_cast<NodeId>(storage_.size()) : free_slots_.back();
    const auto undo = metadata_.write(slot, metadata);
    const int level = random_level();
    NodeId node;
    try {
        if (!free_slots_.empty()) {
            node = storage_.insert_into(slot, id, scratch_.values(), level);
            free_slots_.pop_back();
        } else {
            node = storage_.insert(id, scratch_.values(), level);
        }
    } catch (...) {
        metadata_.undo_write(slot, undo);
        throw;
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
        // like any other vector: free its ID and put its slot back on the free
        // list. Undo the metadata write fully, including any field or string it
        // created (linking never touches metadata, so this write is the latest).
        storage_.ids().release(id);
        metadata_.undo_write(node, undo);
        --live_;
        free_slots_.push_back(node);  // cannot throw: capacity reserved in step 1
        throw;
    }
}

void HnswIndex::link_new_node(NodeId node, int level, NodeId ignore) {
    // `ignore` (the old version of an updated vector) is traversed but never
    // chosen as a neighbor, since it is about to be removed.
    const Eligibility eligible{&storage_.ids(), nullptr, nullptr, ignore};
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
            search_level(query, entries, params_.ef_construction, l, *visited, node, eligible);
        if (found.empty()) {
            // Every node reachable from here is removed (possible after many
            // removals without repair). Search again from the entry point too,
            // which is always live, so the new node links to live nodes.
            entries.push_back({distance_to(query, entry_), entry_});
            found = search_level(query, entries, params_.ef_construction, l, *visited, node, eligible);
        }
        connect(node, select_neighbors(found, params_.M), l);
        if (!found.empty()) entries = std::move(found);  // empty only when updating the only vector
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
            // Coverage: nb == node is impossible (lists never contain self-links).
            if (nb != node && !ids.is_deleted(nb) && graph.level(nb) >= l) around.push_back(nb);  // GCOVR_EXCL_BR_LINE
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
    metadata_.clear(*node);
    --live_;
    free_slots_.push_back(*node);
    if (*node == entry_) choose_new_entry();
    return true;
}

bool HnswIndex::update(std::uint64_t id, std::span<const float> vector) { return replace(id, vector, nullptr); }

bool HnswIndex::update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    return replace(id, vector, &metadata);
}

bool HnswIndex::upsert(std::uint64_t id, std::span<const float> vector) {
    if (replace(id, vector, nullptr)) return false;
    add(id, vector);
    return true;
}

bool HnswIndex::upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    if (replace(id, vector, &metadata)) return false;
    add(id, vector, metadata);
    return true;
}

bool HnswIndex::replace(std::uint64_t id, std::span<const float> vector, const Metadata* metadata) {
    // 1. Validate everything before changing anything.
    scratch_.prepare(vector, metric_);
    if (metadata) metadata_.validate(*metadata);
    IdMap& ids = storage_.ids();
    const auto found = ids.find(id);
    if (!found) return false;
    const NodeId old = *found;
    reserve_one_more(free_slots_);  // room for the old slot (or an abandoned new one)

    // 2. Build the new version in another slot. The old one stays live and
    //    findable throughout, so any failure below leaves it untouched.
    const Metadata md = metadata ? *metadata : metadata_.read(old);
    const bool reuse = !free_slots_.empty();
    const NodeId slot = reuse ? free_slots_.back() : static_cast<NodeId>(storage_.size());
    const auto undo = metadata_.write(slot, md);
    const int level = random_level();
    try {
        if (reuse) storage_.insert_into_unmapped(slot, id, scratch_.values(), level);
        else storage_.insert_unmapped(id, scratch_.values(), level);
    } catch (...) {
        metadata_.undo_write(slot, undo);
        throw;
    }

    // 3. Link the new version (never to the old one), then repair the graph
    //    around the old one while it is still live, as remove() does.
    const NodeId saved_entry = entry_;
    const int saved_max_level = max_level_;
    try {
        link_new_node(slot, level, old);
        if (params_.repair_on_remove) repair_neighbors(old);
    } catch (...) {
        // Abandon the new version like a failed insert. Links made so far are
        // valid; the repair only re-chose neighbors' links.
        ids.retire(slot);
        metadata_.undo_write(slot, undo);
        entry_ = saved_entry;
        max_level_ = saved_max_level;
        if (!reuse) free_slots_.push_back(slot);  // cannot throw: reserved in step 1
        throw;
    }

    // 4. Commit: the ID now finds the new version; the old slot is freed.
    //    Nothing below can throw.
    ids.repoint(id, slot);
    ids.retire(old);
    metadata_.clear(old);
    if (reuse) free_slots_.pop_back();
    free_slots_.push_back(old);
    if (old == entry_) choose_new_entry();
    return true;
}

bool HnswIndex::contains(std::uint64_t id) const { return storage_.ids().find(id).has_value(); }

bool HnswIndex::set_metadata(std::uint64_t id, const Metadata& metadata) {
    const auto node = storage_.ids().find(id);
    if (!node) return false;
    metadata_.update(*node, metadata);
    return true;
}

std::optional<Metadata> HnswIndex::get_metadata(std::uint64_t id) const {
    const auto node = storage_.ids().find(id);
    if (!node) return std::nullopt;
    return metadata_.read(*node);
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k,
                                            std::size_t ef) const {
    return search(query, k, ef, SearchOptions{});
}

Eligibility HnswIndex::eligibility(const SearchOptions& options, std::optional<CompiledFilter>& holder) const {
    if (options.planner) validate(*options.planner);
    if (options.filter && !options.filter->is_all()) holder.emplace(*options.filter, metadata_);
    return Eligibility{&storage_.ids(), holder ? &*holder : nullptr,
                       options.predicate ? &options.predicate : nullptr};
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k, std::size_t ef,
                                            const SearchOptions& options) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);  // validates even when the index is empty
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    SearchStats stats;
    auto results = search_impl(q.data(), k, ef, eligible, options.strategy,
                               options.planner ? *options.planner : planner_, stats);
    if (options.stats) *options.stats = stats;
    return results;
}

std::vector<Candidate> HnswIndex::exact_scan(const float* query, std::size_t k, const Eligibility& eligible,
                                             SearchStats& stats) const {
    TopK top(k);
    for (NodeId n = 0; n < storage_.size(); ++n) {
        if (!eligible(n)) continue;
        ++stats.distance_computations;
        top.push({distance_to(query, n), n});
    }
    stats.nodes_visited += storage_.size();
    return top.take_sorted();
}

std::vector<SearchResult> HnswIndex::search_impl(const float* query, std::size_t k, std::size_t ef,
                                                 const Eligibility& eligible, Strategy strategy,
                                                 const PlannerParams& params, SearchStats& stats) const {
    if (k == 0 || live_ == 0) return {};
    k = std::min(k, live_);                          // cannot return more than exists
    ef = std::min(std::max(ef, k), storage_.size());  // at least k, at most every node

    // Plan: no filter means a plain graph search unless exact search is forced.
    Strategy run = strategy == Strategy::ForceExact ? Strategy::ForceExact : Strategy::ForceGraph;
    stats.reason = strategy == Strategy::Auto ? PlanReason::NoFilter : PlanReason::Forced;
    if (eligible.filtered()) {
        const std::optional<std::size_t> counted =
            (eligible.filter && !eligible.predicate) ? eligible.filter->exact_count() : std::nullopt;
        const PlanDecision d = plan_search(storage_.size(), live_, eligible, counted, ef, strategy, params);
        run = d.strategy;
        stats.reason = d.reason;
        stats.selectivity = d.selectivity;
        stats.estimated_matches = d.estimated_matches;
        stats.exact_count = d.exact_count;
        ef = std::max(ef, d.ef);
    }
    stats.strategy = run;

    std::vector<Candidate> found;
    if (run == Strategy::ForceExact) {
        found = exact_scan(query, k, eligible, stats);
    } else {
        stats.ef = ef;
        // Greedy descent from the entry point to level 1.
        Candidate current{distance_to(query, entry_), entry_};
        for (int l = max_level_; l > 0; --l) current = greedy_closest(query, current, l, kEmpty);

        // Beam search on level 0, returning only eligible nodes.
        auto visited = visited_pool_.acquire(storage_.size());
        found = search_level(query, {current}, ef, 0, *visited, kEmpty, eligible, &stats);
        if (found.size() < k) {
            // The descent ended where few eligible nodes are reachable (possible
            // after many removals without repair, or with a selective filter; links
            // are one-directional). Search again from the entry point too, which is
            // always live.
            found = search_level(query, {current, {distance_to(query, entry_), entry_}}, ef, 0,
                                 *visited, kEmpty, eligible, &stats);
        }
    }
    if (found.size() > k) found.resize(k);

    std::vector<SearchResult> results;
    results.reserve(found.size());
    for (const Candidate& c : found) results.push_back({storage_.ids().external(c.id), c.distance});
    return results;
}

std::vector<std::vector<SearchResult>> HnswIndex::search_batch(std::span<const float> queries, std::size_t k,
                                                               std::size_t ef, std::size_t threads,
                                                               const SearchOptions& options) const {
    const std::size_t nq = validate_batch(queries, dim());  // every query checked before any work
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    const PlannerParams& params = options.planner ? *options.planner : planner_;
    std::vector<std::vector<SearchResult>> results(nq);
    auto one = [&](std::size_t i) {
        PreparedVector q(dim());
        q.prepare(queries.subspan(i * dim(), dim()), metric_);
        SearchStats stats;
        results[i] = search_impl(q.data(), k, ef, eligible, options.strategy, params, stats);
    };
    if (threads == 1 || nq <= 1) {
        for (std::size_t i = 0; i < nq; ++i) one(i);
    } else {
        pool_.get(threads)->parallel_for(nq, one);
    }
    return results;
}

std::vector<SearchResult> HnswIndex::search_range(std::span<const float> query, float radius,
                                                  std::size_t max_results, const SearchOptions& options) const {
    PreparedVector q(dim());
    q.prepare(query, metric_);
    if (std::isnan(radius)) throw std::invalid_argument("radius is NaN");
    std::optional<CompiledFilter> compiled;
    const Eligibility eligible = eligibility(options, compiled);
    SearchStats stats;
    std::vector<Candidate> hits;

    if (max_results > 0 && live_ > 0) {
        if (options.strategy == Strategy::ForceExact) {
            stats.strategy = Strategy::ForceExact;
            stats.reason = PlanReason::Forced;
            for (NodeId n = 0; n < storage_.size(); ++n) {
                if (!eligible(n)) continue;
                ++stats.distance_computations;
                const float d = distance_to(q.data(), n);
                if (d <= radius) hits.push_back({d, n});
            }
            stats.nodes_visited = storage_.size();
        } else {
            stats.strategy = Strategy::ForceGraph;
            stats.reason = options.strategy == Strategy::Auto ? PlanReason::Graph : PlanReason::Forced;
            // 1. Seeds: the nearest live vectors, ignoring the filter (vectors that
            //    fail it still connect the region).
            const std::size_t seeds_ef = std::min<std::size_t>(storage_.size(), 64);
            Candidate current{distance_to(q.data(), entry_), entry_};
            for (int l = max_level_; l > 0; --l) current = greedy_closest(q.data(), current, l, kEmpty);
            //    The entry point (always live) is always a starting point too: links
            //    are one-directional, so after many removals the descent can end in
            //    a region that reaches only some of the live vectors.
            auto visited = visited_pool_.acquire(storage_.size());
            std::vector<Candidate> starts{current};
            if (current.id != entry_) starts.push_back({distance_to(q.data(), entry_), entry_});
            std::vector<Candidate> seeds =
                search_level(q.data(), starts, seeds_ef, 0, *visited, kEmpty, live_only(), &stats);

            // 2. Grow the region: from every vector inside the radius, visit its
            //    neighbors; vectors outside may be crossed for up to
            //    `range_expand_outside` hops to reach more of the region.
            visited->reset(storage_.size());
            std::vector<std::pair<NodeId, std::size_t>> queue;  // (node, hops outside the radius)
            auto consider = [&](NodeId n, float d, std::size_t parent_hops) {
                const bool inside = d <= radius;
                if (inside && eligible(n)) hits.push_back({d, n});
                const std::size_t hops = inside ? 0 : parent_hops + 1;
                if (hops <= options.range_expand_outside) queue.push_back({n, hops});
            };
            for (const Candidate& s : seeds)
                if (visited->visit(s.id)) consider(s.id, s.distance, 0);  // GCOVR_EXCL_BR_LINE: seeds are unique
            const GraphStorage& graph = storage_.graph();
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const auto [node, hops] = queue[head];
                ++stats.nodes_visited;
                for (NodeId nb : graph.links(node, 0)) {
                    if (nb == kEmpty) break;
                    if (!visited->visit(nb)) continue;
                    ++stats.distance_computations;
                    consider(nb, distance_to(q.data(), nb), hops);
                }
            }
        }
        std::sort(hits.begin(), hits.end());
        if (hits.size() > max_results) hits.resize(max_results);
    }
    if (options.stats) *options.stats = stats;
    std::vector<SearchResult> results;
    results.reserve(hits.size());
    for (const Candidate& c : hits) results.push_back({storage_.ids().external(c.id), c.distance});
    return results;
}

CompactStats HnswIndex::compact() {
    const std::size_t reclaimed = deleted_count();
    // Build the replacement completely first, so a failure leaves this index intact.
    HnswIndex fresh(dim(), metric_, params_, metadata_.current_schema());
    for (const std::string& field : metadata_.indexed_fields()) fresh.metadata_.index_field(field);
    fresh.planner_ = planner_;
    const IdMap& ids = storage_.ids();
    for (NodeId n = 0; n < storage_.size(); ++n)
        if (!ids.is_deleted(n)) fresh.add(ids.external(n), storage_.vectors().get(n), metadata_.read(n));

    // Commit: moves only, nothing can throw.
    storage_ = std::move(fresh.storage_);
    metadata_ = std::move(fresh.metadata_);
    rng_ = fresh.rng_;
    entry_ = fresh.entry_;
    max_level_ = fresh.max_level_;
    live_ = fresh.live_;
    free_slots_ = std::move(fresh.free_slots_);
    visited_pool_.clear();  // pooled lists were sized for the old, larger index
    return {live_, reclaimed};
}

}  // namespace vecdb