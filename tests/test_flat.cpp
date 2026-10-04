/**
 * @file test_flat.cpp
 * @brief Tests for FlatIndex. Prints "All FlatIndex tests passed."
 *
 * Checks that:
 *  - Results exactly match a simple reference (compute every distance, sort,
 *    take k) for all three metrics. The reference uses the same kernel, so
 *    this tests the search logic; the kernels are tested in test_distance.cpp.
 *  - Removed vectors never appear; remove/contains/size behave correctly.
 *  - Cosine works on un-normalized input.
 *  - Edge cases: empty index, k = 0, k larger than the index, wrong dimension,
 *    duplicate IDs (including IDs that were removed).
 */
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "flat_index.h"
#include "check.h"

using namespace vecdb;

static std::vector<std::vector<float>> random_vectors(std::size_t count, std::size_t dim,
                                                      unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> out(count, std::vector<float>(dim));
    for (auto& v : out)
        for (float& x : v) x = dist(rng);
    return out;
}

/// Reference answer: distance to every vector (skipping `removed`), sorted, first k.
static std::vector<SearchResult> reference(const std::vector<std::vector<float>>& data,
                                           const std::vector<bool>& removed,
                                           const std::vector<float>& query, Metric m,
                                           std::size_t k) {
    const std::size_t dim = query.size();
    PreparedVector q(dim), row(dim);
    q.prepare(query, m);
    DistanceFn fn = get_distance(m);
    std::vector<Candidate> all;
    for (NodeId i = 0; i < data.size(); ++i) {
        if (removed[i]) continue;
        row.prepare(data[i], m);
        all.push_back({fn(q.data(), row.data(), q.stride()), i});
    }
    std::sort(all.begin(), all.end());
    std::vector<SearchResult> out;
    for (std::size_t i = 0; i < std::min(k, all.size()); ++i)
        out.push_back({1000 + std::uint64_t(all[i].id), all[i].distance});
    return out;
}

static bool same(const std::vector<SearchResult>& a, const std::vector<SearchResult>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].distance != b[i].distance) return false;
    return true;
}

static void test_matches_reference() {
    const std::size_t dim = 37;  // not a multiple of 16, so padding is exercised
    auto data = random_vectors(500, dim, 1);
    auto queries = random_vectors(20, dim, 2);
    std::vector<bool> removed(data.size(), false);

    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        FlatIndex index(dim, m);
        for (std::size_t i = 0; i < data.size(); ++i) index.add(1000 + i, data[i]);
        CHECK(index.size() == 500);
        for (const auto& q : queries)
            for (std::size_t k : {1, 10, 50})
                CHECK(same(index.search(q, k), reference(data, removed, q, m, k)));
    }
}

static void test_remove() {
    const std::size_t dim = 16;
    auto data = random_vectors(200, dim, 3);
    auto queries = random_vectors(10, dim, 4);
    FlatIndex index(dim, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) index.add(1000 + i, data[i]);

    std::vector<bool> removed(data.size(), false);
    for (std::size_t i = 0; i < data.size(); i += 3) {
        CHECK(index.remove(1000 + i));
        removed[i] = true;
    }
    CHECK(index.size() == 200 - 67);
    CHECK(!index.remove(1000));      // already removed
    CHECK(!index.remove(999999));    // never existed
    CHECK(!index.contains(1000) && index.contains(1001));

    for (const auto& q : queries) {
        auto got = index.search(q, 20);
        CHECK(same(got, reference(data, removed, q, Metric::L2, 20)));
        for (const auto& r : got) CHECK(!removed[r.id - 1000]);
    }
}

static void test_cosine_unnormalized() {
    FlatIndex index(3, Metric::Cosine);
    index.add(1, std::vector<float>{1, 0, 0});
    index.add(2, std::vector<float>{0, 5, 0});  // long vector, same direction as y
    auto r = index.search(std::vector<float>{0, 0.1f, 0}, 2);
    CHECK(r[0].id == 2 && std::abs(r[0].distance) < 1e-6f);  // same direction
    CHECK(r[1].id == 1 && std::abs(r[1].distance - 1.0f) < 1e-6f);  // perpendicular
}

static void test_edge_cases() {
    FlatIndex index(4, Metric::L2);
    std::vector<float> v{1, 2, 3, 4};
    CHECK(index.search(v, 5).empty());  // empty index
    CHECK(throws([&] { index.search(std::vector<float>(3), 1); }));  // wrong dim, even when empty

    index.add(7, v);
    index.add(8, std::vector<float>{0, 0, 0, 0});
    CHECK(index.search(v, 0).empty());  // k = 0
    auto all = index.search(v, 100);    // k larger than the index
    CHECK(all.size() == 2 && all[0].id == 7 && all[0].distance == 0.0f);
    CHECK(all[0].distance <= all[1].distance);

    CHECK(throws([&] { index.add(7, v); }));                          // duplicate
    CHECK(throws([&] { index.add(9, std::vector<float>(5)); }));      // wrong dim
    CHECK(index.size() == 2 && !index.contains(9));                   // nothing changed
    index.remove(8);
    CHECK(throws([&] { index.add(8, v); }));  // removed IDs cannot be reused
    CHECK(index.search(v, 10).size() == 1);
}

int main() {
    test_matches_reference();
    test_remove();
    test_cosine_unnormalized();
    test_edge_cases();
    std::printf("All FlatIndex tests passed.\n");
}