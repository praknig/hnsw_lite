/**
 * @file upkeep.cpp
 * @brief HnswIndex::upkeep: incremental repair after churn; see health.h.
 */
#include <algorithm>
#include <chrono>

#include "hnsw_index.h"

namespace vecdb {

UpkeepStats HnswIndex::upkeep(const UpkeepOptions& upkeep_options, const OperationOptions& options) {
    const auto guard = lock_.write("upkeep");
    return upkeep_unlocked(upkeep_options, options);
}

UpkeepStats HnswIndex::upkeep_unlocked(const UpkeepOptions& upkeep_options, const OperationOptions& options) {
    UpkeepStats st;
    GraphStorage& graph = storage_.graph();
    const IdMap& ids = storage_.ids();
    const std::size_t n = storage_.size();
    auto stop = [&] {
        if (upkeep_options.max_nodes > 0 && st.nodes_repaired + st.reconnected >= upkeep_options.max_nodes) return true;
        if (options.cancel && options.cancel->load()) return true;
        return options.deadline && std::chrono::steady_clock::now() >= *options.deadline;
    };
    if (live_ == 0) {
        st.complete = true;
        removals_since_upkeep_ = updates_since_upkeep_ = 0;
        return st;
    }

    // 1. Consolidate: replace links to removed vectors by the best of the node's
    //    other links and the removed vectors' live links on that level.
    if (upkeep_options.consolidate) {
        for (NodeId node = 0; node < n; ++node) {
            if (ids.is_deleted(node)) continue;
            bool repaired = false;
            for (int l = 0; l <= graph.level(node); ++l) {
                const auto slots = graph.links(node, l);
                const std::size_t count = GraphStorage::count(slots);
                // Links to removed vectors, and stale links: to a vector without this
                // level, after a removed vector's slot was reused by a lower one (found
                // by uk10: such links were reported by health() but never repaired).
                bool dangling = false;
                for (std::size_t i = 0; i < count && !dangling; ++i) dangling = ids.is_deleted(slots[i]) || graph.level(slots[i]) < l;
                if (!dangling) continue;
                if (stop()) return st;
                std::vector<Candidate> options_l;
                auto consider = [&](NodeId c) {
                    if (c == node || ids.is_deleted(c) || graph.level(c) < l) return;
                    options_l.push_back({distance_between(node, c), c});
                };
                for (std::size_t i = 0; i < count; ++i) {
                    const NodeId c = slots[i];
                    if (!ids.is_deleted(c) && graph.level(c) >= l) {
                        consider(c);
                        continue;
                    }
                    ++st.links_replaced;
                    if (graph.level(c) < l || !ids.is_deleted(c)) continue;  // stale: nothing to borrow on this level
                    for (NodeId c2 : graph.links(c, l)) {
                        if (c2 == kEmpty) break;
                        consider(c2);
                    }
                }
                std::sort(options_l.begin(), options_l.end());
                options_l.erase(std::unique(options_l.begin(), options_l.end(),
                                            [](const Candidate& a, const Candidate& b) { return a.id == b.id; }),
                                options_l.end());
                graph.set_links(node, l, select_neighbors(node, options_l, l == 0 ? graph.M0() : graph.M()));
                repaired = true;
            }
            st.nodes_repaired += repaired;
        }
    }

    // 2. Reconnect: every live vector a search cannot reach (on level 0, through
    //    live vectors only) gets an incoming link from a nearby reachable vector,
    //    found by searching for it: preferably into a free slot, otherwise in place
    //    of a link whose target keeps another incoming link from a reachable
    //    vector. Repeated until all are reachable (at most 16 rounds).
    auto walk = [&](std::vector<bool>& reached) {
        reached.assign(n, false);
        std::vector<NodeId> stack{entry_};
        reached[entry_] = true;
        std::size_t count = 1;
        while (!stack.empty()) {
            const NodeId v = stack.back();
            stack.pop_back();
            for (NodeId nb : graph.links(v, 0)) {
                if (nb == kEmpty) break;
                if (!reached[nb] && !ids.is_deleted(nb)) {
                    reached[nb] = true;
                    stack.push_back(nb);
                    ++count;
                }
            }
        }
        return count;
    };
    std::vector<bool> reached;
    std::size_t reachable = walk(reached);
    if (upkeep_options.reconnect) {
        std::vector<float> buffer;
        const Eligibility live{&ids};
        // Rounds continue while they make progress: each newly reached vector brings its
        // own links, and with them more of the graph.
        while (reachable < live_) {
            std::vector<std::uint32_t> indegree(n, 0);
            for (NodeId v = 0; v < n; ++v)
                if (reached[v])
                    for (NodeId nb : graph.links(v, 0)) {
                        if (nb == kEmpty) break;
                        ++indegree[nb];
                    }
            bool changed = false;
            for (NodeId u = 0; u < n; ++u) {
                if (ids.is_deleted(u) || reached[u]) continue;
                if (stop()) return st;
                const float* q = stored_padded(u, buffer).data();
                Candidate current{distance_to(q, entry_), entry_};
                for (int l = max_level_; l > 0; --l) current = greedy_closest(q, current, l, kEmpty);
                auto visited = visited_pool_.acquire(n);
                Eligibility others = live;
                others.exclude = u;
                const auto hosts = search_level(q, {current}, params_.ef_construction, 0, *visited, kEmpty, others, nullptr);
                bool linked = false;
                for (const Candidate& h : hosts) {  // a free slot, nearest first
                    if (!reached[h.id]) continue;
                    const auto slots = graph.links(h.id, 0);
                    const std::size_t count = GraphStorage::count(slots);
                    if (count < slots.size()) {
                        slots[count] = u;
                        linked = true;
                        break;
                    }
                }
                for (std::size_t i = 0; !linked && i < hosts.size(); ++i) {  // else a link that is not the target's only one
                    if (!reached[hosts[i].id]) continue;
                    const auto slots = graph.links(hosts[i].id, 0);
                    for (std::size_t s = GraphStorage::count(slots); s-- > 0;)
                        if (indegree[slots[s]] >= 2) {
                            --indegree[slots[s]];
                            slots[s] = u;
                            linked = true;
                            break;
                        }
                }
                // Else the nearest reachable vector with a free slot, found by a scan: when the
                // reachable part is tiny, a search finds only a few hosts, soon full (found
                // with many identical vectors and 4 links per vector: 440 of 480 stayed
                // unreachable). Only here does upkeep pay for a pass over the reachable vectors.
                if (!linked) {
                    NodeId best = kEmpty;
                    float best_d = 0;
                    for (NodeId v = 0; v < n; ++v) {
                        if (!reached[v] || v == u) continue;
                        const auto slots = graph.links(v, 0);
                        if (GraphStorage::count(slots) >= slots.size()) continue;
                        const float d = distance_to(q, v);
                        if (best == kEmpty || d < best_d) {
                            best = v;
                            best_d = d;
                        }
                    }
                    if (best != kEmpty) {
                        const auto slots = graph.links(best, 0);
                        slots[GraphStorage::count(slots)] = u;
                        linked = true;
                    }
                }
                if (!linked) continue;  // GCOVR_EXCL_LINE: every reachable vector's list is full, and every link its target's only one
                ++indegree[u];
                ++st.reconnected;
                changed = true;
            }
            reachable = walk(reached);
            if (!changed) break;  // GCOVR_EXCL_LINE: nothing could be linked
        }
    }
    st.unreachable = live_ - reachable;
    st.complete = true;
    removals_since_upkeep_ = updates_since_upkeep_ = 0;
    return st;
}

}  // namespace vecdb