/**
 * @file test_hnsw.cpp
 * @brief Tests for HnswIndex. Prints "All HnswIndex tests passed."
 *
 * Checks that:
 *  - The graph is valid: links point to real nodes on that level, no self-links,
 *    no duplicates, no list over capacity, entry point on the top level, and
 *    almost every node reachable from the entry point on level 0.
 *  - Recall@10 against FlatIndex is high for all three metrics.
 *  - Removed vectors never appear, and recall stays high after removals.
 *  - The same seed builds the same graph and gives the same results.
 *  - Edge cases: empty index, one vector, k = 0, k > size, ef < k, wrong
 *    dimension, duplicate IDs, invalid parameters.
 */
#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "flat_index.h"
#include "hnsw_index.h"
#include "check.h"

using namespace vecdb;

/// Points scattered around `clusters` random centers (similar to real embeddings).
static std::vector<std::vector<float>> clustered(std::size_t count, std::size_t dim,
                                                 std::size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (float& x : c) x = center(rng);
    std::vector<std::vector<float>> out(count, std::vector<float>(dim));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t d = 0; d < dim; ++d) out[i][d] = centers[rng() % clusters][d] + noise(rng);
    return out;
}

/// Average fraction of the true top-k (from FlatIndex) that HNSW also returned.
static double recall(const HnswIndex& hnsw, const FlatIndex& flat,
                     const std::vector<std::vector<float>>& queries, std::size_t k,
                     std::size_t ef) {
    double total = 0.0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : flat.search(q, k)) truth.insert(r.id);
        std::size_t hits = 0;
        for (const auto& r : hnsw.search(q, k, ef)) hits += truth.count(r.id);
        total += double(hits) / double(truth.size());
    }
    return total / double(queries.size());
}

/// Checks every structural rule of the graph.
static void check_graph(const HnswIndex& index) {
    const Storage& st = index.storage();
    const GraphStorage& g = st.graph();
    const std::size_t n = st.size();

    CHECK(index.entry_point() < n);
    CHECK(g.level(index.entry_point()) == index.max_level());

    for (NodeId node = 0; node < n; ++node) {
        CHECK(g.level(node) <= index.max_level());
        for (int level = 0; level <= g.level(node); ++level) {
            auto slots = g.links(node, level);
            const std::size_t count = GraphStorage::count(slots);
            CHECK(count <= (level == 0 ? g.M0() : g.M()));
            std::set<NodeId> seen;
            for (std::size_t i = 0; i < count; ++i) {
                NodeId nb = slots[i];
                CHECK(nb < n && nb != node);         // real node, no self-link
                CHECK(g.level(nb) >= level);         // neighbor exists on this level
                CHECK(seen.insert(nb).second);       // no duplicates
            }
            for (std::size_t i = count; i < slots.size(); ++i) CHECK(slots[i] == kEmpty);
        }
    }

    // Level 0 connectivity: walk all links from the entry point.
    std::vector<bool> reached(n, false);
    std::vector<NodeId> stack{index.entry_point()};
    reached[index.entry_point()] = true;
    std::size_t count = 1;
    while (!stack.empty()) {
        NodeId node = stack.back();
        stack.pop_back();
        for (NodeId nb : g.links(node, 0)) {
            if (nb == kEmpty) break;
            if (!reached[nb]) {
                reached[nb] = true;
                ++count;
                stack.push_back(nb);
            }
        }
    }
    CHECK(double(count) / double(n) >= 0.99);
}

static void test_recall_all_metrics() {
    const std::size_t dim = 32, n = 3000;
    auto data = clustered(n, dim, 20, 1);
    auto queries = clustered(100, dim, 20, 2);

    for (Metric m : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        HnswIndex hnsw(dim, m);
        FlatIndex flat(dim, m);
        for (std::size_t i = 0; i < n; ++i) {
            hnsw.add(i, data[i]);
            flat.add(i, data[i]);
        }
        CHECK(hnsw.size() == n);
        check_graph(hnsw);

        const double r = recall(hnsw, flat, queries, 10, 100);
        std::printf("  %-13s recall@10 (ef=100): %.3f\n",
                    m == Metric::L2 ? "L2" : m == Metric::Cosine ? "Cosine" : "InnerProduct", r);
        CHECK(r >= 0.95);
    }
}

static void test_results_sorted_and_exact_match() {
    const std::size_t dim = 16;
    auto data = clustered(500, dim, 5, 3);
    HnswIndex index(dim, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) index.add(i, data[i]);

    for (std::size_t i = 0; i < 20; ++i) {
        auto r = index.search(data[i], 10, 50);
        CHECK(r.size() == 10);
        CHECK(r[0].id == i && r[0].distance == 0.0f);  // a stored vector finds itself
        for (std::size_t j = 1; j < r.size(); ++j) CHECK(r[j - 1].distance <= r[j].distance);
    }
}

static void test_remove() {
    const std::size_t dim = 32, n = 2000;
    auto data = clustered(n, dim, 10, 4);
    auto queries = clustered(50, dim, 10, 5);
    HnswIndex hnsw(dim, Metric::L2);
    FlatIndex flat(dim, Metric::L2);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(i, data[i]);
        flat.add(i, data[i]);
    }

    // Remove every 4th vector, including the entry point.
    const std::uint64_t entry_id = hnsw.storage().ids().external(hnsw.entry_point());
    for (std::size_t i = 0; i < n; i += 4) {
        CHECK(hnsw.remove(i));
        flat.remove(i);
    }
    if (entry_id % 4 != 0) {
        CHECK(hnsw.remove(entry_id));
        flat.remove(entry_id);
    }
    CHECK(!hnsw.contains(entry_id) && !hnsw.remove(entry_id));
    CHECK(hnsw.size() == flat.size());

    for (const auto& q : queries)
        for (const auto& r : hnsw.search(q, 10, 100)) CHECK(hnsw.contains(r.id));
    CHECK(recall(hnsw, flat, queries, 10, 100) >= 0.95);

    // Inserts after removals still work.
    hnsw.add(n + 1, data[0]);
    CHECK(hnsw.search(data[0], 1)[0].id == n + 1);
}

static void test_deterministic() {
    const std::size_t dim = 16;
    auto data = clustered(800, dim, 8, 6);
    HnswIndex a(dim, Metric::L2), b(dim, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        a.add(i, data[i]);
        b.add(i, data[i]);
    }
    CHECK(a.entry_point() == b.entry_point() && a.max_level() == b.max_level());
    for (std::size_t i = 0; i < 20; ++i) {
        auto ra = a.search(data[i * 7], 10), rb = b.search(data[i * 7], 10);
        CHECK(ra.size() == rb.size());
        for (std::size_t j = 0; j < ra.size(); ++j)
            CHECK(ra[j].id == rb[j].id && ra[j].distance == rb[j].distance);
    }
}

static void test_edge_cases() {
    HnswIndex index(4, Metric::L2);
    std::vector<float> v{1, 2, 3, 4};
    CHECK(index.search(v, 5).empty() && index.max_level() == -1 && index.entry_point() == kEmpty);
    CHECK(throws([&] { index.search(std::vector<float>(3), 1); }));

    index.add(7, v);  // single vector
    auto one = index.search(std::vector<float>{0, 0, 0, 0}, 3);
    CHECK(one.size() == 1 && one[0].id == 7);
    CHECK(index.entry_point() == 0);

    for (std::uint64_t i = 0; i < 5; ++i)
        index.add(100 + i, std::vector<float>{float(i), 0, 0, 0});
    CHECK(index.search(v, 0).empty());                // k = 0
    CHECK(index.search(v, 100, 1).size() == 6);       // k > size, ef < k raised to k

    CHECK(throws([&] { index.add(7, v); }));                      // duplicate
    CHECK(throws([&] { index.add(9, std::vector<float>(5)); }));  // wrong dim
    CHECK(index.size() == 6 && !index.contains(9));

    index.remove(7);
    for (std::uint64_t i = 0; i < 5; ++i) index.remove(100 + i);
    CHECK(index.size() == 0 && index.search(v, 3).empty());  // everything removed

    CHECK(throws([] { HnswIndex bad(4, Metric::L2, HnswParams{1, 200, 42}); }));  // M < 2
    CHECK(throws([] { HnswIndex bad(4, Metric::L2, HnswParams{16, 0, 42}); }));   // ef_c = 0
}

static void test_level_distribution() {
    // About 1 in M nodes should reach level 1 or above.
    HnswIndex index(16, Metric::L2, HnswParams{16, 32, 7});
    auto data = clustered(4000, 16, 4, 8);
    for (std::size_t i = 0; i < data.size(); ++i) index.add(i, data[i]);
    std::size_t upper = 0;
    for (NodeId node = 0; node < 4000; ++node) upper += index.storage().graph().level(node) > 0;
    const double fraction = double(upper) / 4000.0;  // expected 1/16 = 0.0625
    CHECK(fraction > 0.045 && fraction < 0.08);
    CHECK(index.max_level() >= 1);
}

int main() {
    test_recall_all_metrics();
    test_results_sorted_and_exact_match();
    test_remove();
    test_deterministic();
    test_edge_cases();
    test_level_distribution();
    std::printf("All HnswIndex tests passed.\n");
}