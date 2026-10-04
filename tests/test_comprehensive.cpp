/**
 * @file test_comprehensive.cpp
 * @brief Edge cases, negative scenarios and end-to-end flows across all layers.
 *
 * The per-layer test files check normal behavior. This file pushes every class
 * to its limits and checks that invalid input is rejected with the right
 * exception type and leaves everything unchanged.
 *
 * Sections:
 *  - Layer 1: round_up, AlignedBlock, Arena, VectorStore, IdMap, GraphStorage, Storage
 *  - Layer 2: kernels on extreme values, unaligned input, symmetry; normalize; dispatch
 *  - Layer 3: TopK, PreparedVector, VisitedList(Pool), FlatIndex, HnswIndex
 *  - Concurrency: many threads searching the same index at once
 *  - End to end: full add / search / remove / add lifecycle for every metric
 *
 * Prints each section as it passes, then the total number of checks.
 */
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "arena.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "storage.h"
#include "check.h"

using namespace vecdb;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr std::size_t kHuge = std::numeric_limits<std::size_t>::max();
using Vectors = std::vector<std::vector<float>>;

static const Isa kIsas[] = {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon};
static const Metric kMetrics[] = {Metric::L2, Metric::InnerProduct, Metric::Cosine};

/// Prints a passed section.
static void passed(const char* name) { std::printf("  ok  %s\n", name); }

/// True if `p` is 64-byte aligned.
static bool aligned(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kAlign == 0; }

/// Uniform random vectors in [-1, 1].
static Vectors random_vectors(std::size_t count, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Vectors out(count, std::vector<float>(dim));
    for (auto& v : out)
        for (float& x : v) x = dist(rng);
    return out;
}

/// Points scattered around random centers (behaves like real embeddings).
static Vectors clustered(std::size_t count, std::size_t dim, std::size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    Vectors centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (float& x : c) x = center(rng);
    Vectors out(count, std::vector<float>(dim));
    for (auto& v : out) {
        const auto& c = centers[rng() % clusters];
        for (std::size_t d = 0; d < dim; ++d) v[d] = c[d] + noise(rng);
    }
    return out;
}

/// Average recall@k of `hnsw` against the exact `flat` results.
static double recall(const HnswIndex& hnsw, const FlatIndex& flat, const Vectors& queries,
                     std::size_t k, std::size_t ef) {
    double total = 0.0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : flat.search(q, k)) truth.insert(r.id);
        if (truth.empty()) continue;
        std::size_t hits = 0;
        for (const auto& r : hnsw.search(q, k, ef)) hits += truth.count(r.id);
        total += double(hits) / double(truth.size());
    }
    return total / double(queries.size());
}

/// Checks that results are sorted, unique and only contain live IDs.
template <class Index>
static void check_results(const Index& index, const std::vector<SearchResult>& results) {
    std::set<std::uint64_t> seen;
    for (std::size_t i = 0; i < results.size(); ++i) {
        CHECK(seen.insert(results[i].id).second);  // no duplicate IDs
        CHECK(index.contains(results[i].id));      // never a removed or unknown ID
        if (i > 0) CHECK(results[i - 1].distance <= results[i].distance);
    }
}

/// Checks every structural rule of an HNSW graph, and that at least
/// `min_reachable` of the live nodes can be reached from the entry point.
static void check_graph(const HnswIndex& index, double min_reachable) {
    const Storage& st = index.storage();
    const GraphStorage& g = st.graph();
    const std::size_t n = st.size();
    if (n == 0) {
        CHECK(index.entry_point() == kEmpty && index.max_level() == -1);
        return;
    }
    CHECK(index.entry_point() < n);
    CHECK(g.level(index.entry_point()) == index.max_level());
    for (NodeId node = 0; node < n; ++node) {
        for (int level = 0; level <= g.level(node); ++level) {
            auto slots = g.links(node, level);
            const std::size_t count = GraphStorage::count(slots);
            CHECK(count <= (level == 0 ? g.M0() : g.M()));
            std::set<NodeId> seen;
            for (std::size_t i = 0; i < count; ++i) {
                CHECK(slots[i] < n && slots[i] != node);
                CHECK(g.level(slots[i]) >= level);
                CHECK(seen.insert(slots[i]).second);
            }
            for (std::size_t i = count; i < slots.size(); ++i) CHECK(slots[i] == kEmpty);
        }
    }
    // Reachability of LIVE nodes from the entry point on level 0. The walk may pass
    // through removed nodes; removed nodes themselves need not be reachable.
    std::vector<bool> reached(n, false);
    std::vector<NodeId> stack{index.entry_point()};
    reached[index.entry_point()] = true;
    while (!stack.empty()) {
        NodeId node = stack.back();
        stack.pop_back();
        for (NodeId nb : g.links(node, 0)) {
            if (nb == kEmpty) break;
            if (!reached[nb]) {
                reached[nb] = true;
                stack.push_back(nb);
            }
        }
    }
    std::size_t live = 0, live_reached = 0;
    for (NodeId node = 0; node < n; ++node) {
        if (st.ids().is_deleted(node)) continue;
        ++live;
        live_reached += reached[node];
    }
    if (live > 0) CHECK(double(live_reached) / double(live) >= min_reachable);
}

// ---------------------------------------------------------------------------
// Layer 1
// ---------------------------------------------------------------------------

static void test_round_up_and_constants() {
    CHECK(round_up(0, 16) == 0);
    CHECK(round_up(1, 16) == 16);
    CHECK(round_up(15, 16) == 16);
    CHECK(round_up(16, 16) == 16);
    CHECK(round_up(17, 16) == 32);
    CHECK(round_up(100, 16) == 112);
    CHECK(round_up(64, 64) == 64);
    CHECK(kFloatsPerLine == 16 && kAlign == 64);
    CHECK(kEmpty == std::numeric_limits<std::uint32_t>::max());
    passed("Layer 1: round_up and constants");
}

static void test_aligned_block() {
    AlignedBlock empty(0);  // zero bytes is allowed
    CHECK(empty.size() == 0);

    AlignedBlock ff(10, 0xFF);  // custom fill, size rounded to one cache line
    CHECK(ff.size() == 64 && aligned(ff.data()));
    for (std::size_t i = 0; i < ff.size(); ++i) CHECK(ff.data()[i] == std::byte{0xFF});

    AlignedBlock a(128);
    std::memset(a.data(), 7, 128);
    AlignedBlock b(64);
    b = std::move(a);  // move assignment frees b's old memory, takes a's
    CHECK(a.data() == nullptr && a.size() == 0);
    CHECK(b.size() == 128 && b.data()[127] == std::byte{7});

    AlignedBlock& alias = b;
    b = std::move(alias);  // self-move must keep the data
    CHECK(b.size() == 128 && b.data()[0] == std::byte{7});
    passed("Layer 1: AlignedBlock (zero size, fill, move, self-move)");
}

static void test_arena() {
    Arena a(256);
    for (std::size_t align : {1, 2, 4, 8, 16, 32, 64}) {
        void* p = a.allocate(3, align);  // odd size forces realignment each time
        CHECK(reinterpret_cast<std::uintptr_t>(p) % align == 0);
    }

    Arena exact(256);
    exact.allocate(256, 64);  // fills the first block exactly
    CHECK(exact.block_count() == 1);
    exact.allocate(1);        // one more byte needs a new block
    CHECK(exact.block_count() == 2);

    Arena zero(64);
    CHECK(zero.allocate(0) != nullptr);  // zero bytes still returns a valid address

    Arena big(128);
    void* huge = big.allocate(1000);  // larger than a block: gets its own block
    void* small = big.allocate(8);
    CHECK(huge != nullptr && small != nullptr && huge != small);

    // Many arrays across many blocks: none overlap, none move.
    Arena many(1024);
    std::vector<int*> arrays;
    for (int i = 0; i < 100; ++i) {
        int* p = many.allocate_array<int>(50);
        std::fill_n(p, 50, i);
        arrays.push_back(p);
    }
    CHECK(many.block_count() > 1);
    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 50; ++j) CHECK(arrays[i][j] == i);

    double* d = many.allocate_array<double>(3);
    CHECK(reinterpret_cast<std::uintptr_t>(d) % alignof(double) == 0);
    passed("Layer 1: Arena (alignments, exact fill, zero, oversized, stability)");
}

static void test_vector_store() {
    const std::size_t dims[][2] = {{1, 16}, {16, 16}, {17, 32}, {31, 32}, {32, 32}, {33, 48}};
    for (const auto& d : dims) CHECK(VectorStore(d[0]).stride() == d[1]);

    CHECK(throws_as<std::invalid_argument>([] { VectorStore bad(0); }));

    VectorStore empty(3);
    CHECK(empty.size() == 0);
    CHECK(throws_as<std::out_of_range>([&] { empty.get(0); }));

    VectorStore one_per_shelf(3, /*shelf_bits=*/0);  // every vector on its own shelf
    for (int i = 0; i < 5; ++i) one_per_shelf.add(std::vector<float>{float(i), 0, 0});
    for (NodeId i = 0; i < 5; ++i) {
        CHECK(one_per_shelf.get(i)[0] == float(i));
        CHECK(aligned(one_per_shelf.get(i).data()));
    }

    VectorStore s(4);
    CHECK(throws_as<std::invalid_argument>([&] { s.add(std::vector<float>(3)); }));
    CHECK(throws_as<std::invalid_argument>([&] { s.add(std::vector<float>(5)); }));
    CHECK(s.size() == 0);  // failed adds change nothing

    // Layer 1 is raw storage: any float, including special values, round-trips exactly.
    const float max = std::numeric_limits<float>::max();
    const float denorm = std::numeric_limits<float>::denorm_min();
    NodeId id = s.add(std::vector<float>{-0.0f, max, denorm, kNaN});
    auto v = s.get(id);
    CHECK(std::signbit(v[0]) && v[0] == 0.0f);
    CHECK(v[1] == max && v[2] == denorm && std::isnan(v[3]));
    CHECK(s.get(id).size() == 4 && s.get_padded(id).size() == 16);
    passed("Layer 1: VectorStore (strides, 1 row per shelf, errors, special values)");
}

static void test_id_map() {
    IdMap m;
    const std::uint64_t max_id = std::numeric_limits<std::uint64_t>::max();
    CHECK(m.add(0) == 0 && m.add(max_id) == 1);  // smallest and largest user IDs
    CHECK(*m.find(0) == 0 && *m.find(max_id) == 1);
    CHECK(m.external(1) == max_id);
    CHECK(!m.find(12345).has_value());

    CHECK(throws_as<std::out_of_range>([&] { m.external(99); }));
    CHECK(throws_as<std::out_of_range>([&] { m.mark_deleted(99); }));
    CHECK(throws_as<std::out_of_range>([&] { m.is_deleted(99); }));

    m.mark_deleted(0);
    m.mark_deleted(0);  // twice is harmless
    CHECK(m.is_deleted(0) && m.size() == 2);
    CHECK(throws_as<std::invalid_argument>([&] { m.add(0); }));  // removed IDs stay taken
    passed("Layer 1: IdMap (extreme IDs, unknown IDs, double delete, reuse)");
}

static void test_graph_storage() {
    CHECK(throws_as<std::invalid_argument>([] { GraphStorage bad(0); }));

    GraphStorage g(1);  // smallest M: 1 upper link, 2 level-0 links
    CHECK(g.M() == 1 && g.M0() == 2);
    NodeId top = g.add_node(255);  // highest allowed level
    CHECK(g.level(top) == 255 && g.links(top, 255).size() == 1);
    CHECK(throws_as<std::invalid_argument>([&] { g.add_node(256); }));
    CHECK(throws_as<std::invalid_argument>([&] { g.add_node(-1); }));
    CHECK(g.size() == 1);  // failed adds change nothing

    CHECK(throws_as<std::out_of_range>([&] { g.links(top, -1); }));
    CHECK(throws_as<std::out_of_range>([&] { g.links(99, 0); }));
    CHECK(throws_as<std::out_of_range>([&] { g.level(99); }));

    NodeId n = g.add_node(0);
    std::vector<NodeId> full{0, 5};
    g.set_links(n, 0, full);  // exactly the capacity
    CHECK(GraphStorage::count(g.links(n, 0)) == 2);
    CHECK(throws_as<std::length_error>([&] { g.set_links(n, 0, std::vector<NodeId>{1, 2, 3}); }));
    CHECK(g.links(n, 0)[1] == 5);  // rejected set_links changed nothing
    g.set_links(n, 0, std::vector<NodeId>{});  // empty list clears everything
    CHECK(GraphStorage::count(g.links(n, 0)) == 0);

    const GraphStorage& cg = g;  // read-only access works too
    CHECK(cg.links(top, 0).size() == 2);
    passed("Layer 1: GraphStorage (M=1, level limits, errors, set_links)");
}

static void test_storage() {
    Storage st(4);
    std::vector<float> v{1, 2, 3, 4};
    auto in_sync = [&] {
        return st.ids().size() == st.vectors().size() && st.vectors().size() == st.graph().size();
    };

    CHECK(throws_as<std::invalid_argument>([&] { st.insert(1, std::vector<float>(3), 0); }));
    CHECK(throws_as<std::invalid_argument>([&] { st.insert(1, v, -1); }));
    CHECK(throws_as<std::invalid_argument>([&] { st.insert(1, v, 256); }));
    CHECK(st.size() == 0 && in_sync());  // nothing half-inserted

    NodeId id = st.insert(1, v, 3);
    CHECK(st.graph().level(id) == 3);
    for (int level = 0; level <= 3; ++level) CHECK(GraphStorage::count(st.graph().links(id, level)) == 0);
    CHECK(throws_as<std::invalid_argument>([&] { st.insert(1, v, 0); }));  // duplicate
    CHECK(st.size() == 1 && in_sync());
    passed("Layer 1: Storage (bad dim, bad level, duplicate, stays in sync)");
}

// ---------------------------------------------------------------------------
// Layer 2
// ---------------------------------------------------------------------------

static void test_kernel_edge_values() {
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        auto l2 = get_distance(Metric::L2, isa);
        auto ip = get_distance(Metric::InnerProduct, isa);
        auto cos = get_distance(Metric::Cosine, isa);

        std::vector<float> zero(64, 0.0f);
        CHECK(l2(zero.data(), zero.data(), 0) == 0.0f);  // empty input
        CHECK(ip(zero.data(), zero.data(), 0) == 0.0f);
        CHECK(cos(zero.data(), zero.data(), 0) == 1.0f);
        CHECK(l2(zero.data(), zero.data(), 16) == 0.0f);  // zero vectors
        CHECK(cos(zero.data(), zero.data(), 16) == 1.0f);

        // Overflow to infinity, and NaN propagation (the indexes reject these inputs).
        std::vector<float> a(16, 0.0f), b(16, 0.0f);
        a[0] = 3e38f;
        b[0] = -3e38f;
        CHECK(std::isinf(l2(a.data(), b.data(), 16)));
        a[0] = kNaN;
        for (Metric m : kMetrics) CHECK(std::isnan(get_distance(m, isa)(a.data(), b.data(), 16)));

        // Symmetry and non-negativity on random data, including the leftover loops.
        auto r = random_vectors(2, 4096, 11);
        for (std::size_t n : {16, 48, 80, 4096}) {
            for (Metric m : kMetrics) {
                auto fn = get_distance(m, isa);
                CHECK(fn(r[0].data(), r[1].data(), n) == fn(r[1].data(), r[0].data(), n));
            }
            CHECK(l2(r[0].data(), r[1].data(), n) >= 0.0f);
            CHECK(l2(r[0].data(), r[0].data(), n) == 0.0f);
        }

        // Unaligned input gives exactly the same answer as aligned input.
        std::vector<float> shifted(1 + 64), copy(64);
        auto src = random_vectors(1, 64, 12)[0];
        std::copy(src.begin(), src.end(), shifted.begin() + 1);
        std::copy(src.begin(), src.end(), copy.begin());
        for (Metric m : kMetrics) {
            auto fn = get_distance(m, isa);
            CHECK(fn(shifted.data() + 1, r[0].data(), 64) == fn(copy.data(), r[0].data(), 64));
        }
    }
    passed("Layer 2: kernels (n=0, zeros, overflow, NaN, symmetry, unaligned)");
}

static void test_normalize() {
    std::vector<float> empty;
    normalize(empty);  // nothing to do, must not crash

    std::vector<float> neg{-5.0f};
    normalize(neg);
    CHECK(neg[0] == -1.0f);

    std::vector<float> tiny{1e-40f, 0.0f, 0.0f};  // denormal: 1/length overflows a float
    normalize(tiny);
    CHECK(std::fabs(tiny[0] - 1.0f) < 1e-6f && tiny[1] == 0.0f && tiny[2] == 0.0f);

    std::vector<float> huge{1e30f, 1e30f};
    normalize(huge);
    CHECK(std::fabs(huge[0] - 0.70710678f) < 1e-6f && std::fabs(huge[1] - 0.70710678f) < 1e-6f);

    std::vector<float> zero(8, 0.0f);
    normalize(zero);
    for (float x : zero) CHECK(x == 0.0f);

    for (std::size_t dim = 1; dim <= 100; dim += 9) {
        auto v = random_vectors(1, dim, unsigned(dim))[0];
        normalize(v);
        double len2 = 0.0;
        for (float x : v) len2 += double(x) * x;
        CHECK(std::fabs(len2 - 1.0) < 1e-5);
    }
    passed("Layer 2: normalize (empty, negative, tiny, huge, zero, many dims)");
}

static void test_dispatch() {
    CHECK(isa_supported(Isa::Scalar));
    CHECK(active_isa() == active_isa() && isa_supported(active_isa()));
    for (Metric m : kMetrics) CHECK(get_distance(m) == get_distance(m, active_isa()));
    for (Isa isa : kIsas)
        for (Metric m : kMetrics) CHECK((get_distance(m, isa) != nullptr) == isa_supported(isa));
    CHECK(!(isa_supported(Isa::Neon) && isa_supported(Isa::Avx2)));  // never both
    CHECK(std::strcmp(isa_name(Isa::Scalar), "scalar") == 0);
    CHECK(std::strcmp(isa_name(Isa::Avx2), "avx2") == 0);
    CHECK(std::strcmp(isa_name(Isa::Avx512), "avx512") == 0);
    CHECK(std::strcmp(isa_name(Isa::Neon), "neon") == 0);
    passed("Layer 2: dispatch (stable choice, nullptr iff unsupported, names)");
}

// ---------------------------------------------------------------------------
// Layer 3 helpers
// ---------------------------------------------------------------------------

static void test_topk() {
    TopK one(1);
    for (NodeId i = 0; i < 10; ++i) one.push({float(10 - i), i});
    auto best = one.take_sorted();
    CHECK(best.size() == 1 && best[0].id == 9);
    CHECK(one.take_sorted().empty());  // second take is empty
    one.push({1.0f, 3});               // reusable after take
    CHECK(one.size() == 1);

    TopK ties(2);  // equal distances at the boundary: the smaller id wins
    ties.push({1.0f, 5});
    ties.push({1.0f, 7});
    CHECK(ties.push({1.0f, 6}));   // beats {1, 7}
    CHECK(!ties.push({1.0f, 9}));  // loses to both
    auto t = ties.take_sorted();
    CHECK(t[0].id == 5 && t[1].id == 6);

    TopK inf(1);
    inf.push({kInf, 1});
    inf.push({1.0f, 2});
    CHECK(inf.take_sorted()[0].id == 2);

    TopK huge(kHuge);  // must not try to reserve kHuge entries
    for (NodeId i = 0; i < 3; ++i) huge.push({float(i), i});
    CHECK(huge.size() == 3 && !huge.full());
    passed("Layer 3: TopK (k=1, ties, infinity, huge k, reuse)");
}

static void test_prepared_vector() {
    PreparedVector p(1);
    CHECK(p.stride() == 16);

    PreparedVector q(3);
    q.prepare(std::vector<float>{1, 2, 3}, Metric::L2);
    for (float bad : {kNaN, kInf, -kInf}) {
        CHECK(throws_as<std::invalid_argument>([&] { q.prepare(std::vector<float>{1, bad, 3}, Metric::L2); }));
    }
    CHECK(throws_as<std::invalid_argument>([&] { q.prepare(std::vector<float>{9, 9}, Metric::L2); }));
    CHECK(q.values()[0] == 1.0f && q.values()[1] == 2.0f && q.values()[2] == 3.0f);  // untouched

    q.prepare(std::vector<float>{0, 0, 0}, Metric::Cosine);  // zero stays zero, no NaN
    for (float x : q.values()) CHECK(x == 0.0f);

    q.prepare(std::vector<float>{1e-40f, 0, 0}, Metric::Cosine);  // tiny still normalizes
    CHECK(std::fabs(q.values()[0] - 1.0f) < 1e-6f);

    const float max = std::numeric_limits<float>::max();
    q.prepare(std::vector<float>{max, -max, -0.0f}, Metric::L2);  // finite extremes are fine
    CHECK(q.values()[0] == max && std::signbit(q.values()[2]));
    passed("Layer 3: PreparedVector (NaN/inf rejected, unchanged on error, extremes)");
}

static void test_visited() {
    VisitedList v;
    v.reset(0);
    v.reset(5);  // grows from zero
    CHECK(v.visit(4) && !v.visit(4));
    for (int i = 0; i < 1000; ++i) {  // every search starts clean
        v.reset(5);
        CHECK(!v.visited(4));
        v.visit(4);
    }

    VisitedListPool pool;
    {
        std::vector<VisitedList*> lists;
        auto h1 = pool.acquire(10);
        auto h2 = pool.acquire(20);
        auto h3 = pool.acquire(30);
        lists = {&*h1, &*h2, &*h3};
        CHECK(lists[0] != lists[1] && lists[1] != lists[2]);  // all different
        CHECK(h3->visit(29));                                  // sized correctly
        CHECK(pool.idle_count() == 0);
    }
    CHECK(pool.idle_count() == 3);
    {
        auto h = pool.acquire(100);  // reused list grows for a bigger index
        CHECK(h->visit(99));
    }
    passed("Layer 3: VisitedList and pool (growth, 1000 resets, many handles)");
}

// ---------------------------------------------------------------------------
// FlatIndex
// ---------------------------------------------------------------------------

static void test_flat_index() {
    CHECK(throws_as<std::invalid_argument>([] { FlatIndex bad(0, Metric::L2); }));

    // dim = 1; equal distances are ordered by insertion.
    FlatIndex line(1, Metric::L2);
    line.add(5, std::vector<float>{5});
    line.add(1, std::vector<float>{1});
    line.add(3, std::vector<float>{3});
    auto r = line.search(std::vector<float>{2}, 3);
    CHECK(r.size() == 3 && r[0].id == 1 && r[1].id == 3 && r[2].id == 5);
    CHECK(r[0].distance == 1.0f && r[1].distance == 1.0f && r[2].distance == 9.0f);

    // Identical vectors: all at distance 0, in insertion order.
    FlatIndex same(4, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) same.add(i, std::vector<float>{1, 1, 1, 1});
    r = same.search(std::vector<float>{1, 1, 1, 1}, 10);
    for (std::size_t i = 0; i < 10; ++i) CHECK(r[i].id == i && r[i].distance == 0.0f);
    CHECK(same.search(std::vector<float>{1, 1, 1, 1}, kHuge).size() == 10);  // huge k

    // Invalid input is rejected with the right exception and changes nothing.
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    for (float bad : {kNaN, kInf, -kInf}) {
        CHECK(throws_as<std::invalid_argument>([&] { f.add(2, std::vector<float>{bad, 0}); }));
        CHECK(throws_as<std::invalid_argument>([&] { f.search(std::vector<float>{bad, 0}, 1); }));
    }
    CHECK(throws_as<std::invalid_argument>([&] { f.add(2, std::vector<float>{0}); }));
    CHECK(throws_as<std::invalid_argument>([&] { f.add(1, std::vector<float>{5, 5}); }));
    CHECK(f.size() == 1 && !f.contains(2));
    CHECK(f.search(std::vector<float>{0, 0}, 1)[0].id == 1);

    // Inner product: distances are negative; the largest dot product comes first.
    FlatIndex ip(2, Metric::InnerProduct);
    for (std::uint64_t i = 1; i <= 3; ++i) ip.add(i, std::vector<float>{float(i), 0});
    r = ip.search(std::vector<float>{1, 0}, 3);
    CHECK(r[0].id == 3 && r[0].distance == -3.0f && r[2].id == 1 && r[2].distance == -1.0f);

    // Cosine: zero query or zero stored vector is at distance 1 from everything.
    FlatIndex cos(2, Metric::Cosine);
    cos.add(1, std::vector<float>{3, 4});
    cos.add(2, std::vector<float>{0, 0});
    r = cos.search(std::vector<float>{0, 0}, 2);
    CHECK(r[0].distance == 1.0f && r[1].distance == 1.0f);
    r = cos.search(std::vector<float>{6, 8}, 2);  // same direction, longer
    CHECK(r[0].id == 1 && std::fabs(r[0].distance) < 1e-6f && r[1].id == 2 && r[1].distance == 1.0f);

    // Extreme user IDs work end to end.
    FlatIndex ids(2, Metric::L2);
    const std::uint64_t max_id = std::numeric_limits<std::uint64_t>::max();
    ids.add(0, std::vector<float>{0, 0});
    ids.add(max_id, std::vector<float>{1, 1});
    CHECK(ids.search(std::vector<float>{1, 1}, 1)[0].id == max_id);
    CHECK(ids.remove(max_id) && !ids.remove(max_id) && !ids.contains(max_id));

    // Remove everything, then add again.
    FlatIndex life(2, Metric::L2);
    for (std::uint64_t i = 0; i < 5; ++i) life.add(i, std::vector<float>{float(i), 0});
    for (std::uint64_t i = 0; i < 5; ++i) CHECK(life.remove(i));
    CHECK(life.size() == 0 && life.search(std::vector<float>{0, 0}, 3).empty());
    life.add(100, std::vector<float>{9, 9});
    r = life.search(std::vector<float>{0, 0}, 3);
    CHECK(r.size() == 1 && r[0].id == 100);
    CHECK(life.dim() == 2 && life.metric() == Metric::L2);
    passed("FlatIndex (dim 1, ties, huge k, bad input, IP, cosine zero, IDs, lifecycle)");
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

static void test_hnsw_params_and_input() {
    CHECK(throws_as<std::invalid_argument>([] { HnswIndex bad(0, Metric::L2); }));
    CHECK(throws_as<std::invalid_argument>([] { HnswIndex bad(4, Metric::L2, HnswParams{0, 200, 1}); }));
    CHECK(throws_as<std::invalid_argument>([] { HnswIndex bad(4, Metric::L2, HnswParams{1, 200, 1}); }));
    CHECK(throws_as<std::invalid_argument>([] { HnswIndex bad(4, Metric::L2, HnswParams{16, 0, 1}); }));

    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    h.add(2, std::vector<float>{1, 1});
    const NodeId entry = h.entry_point();
    const int top = h.max_level();
    for (float bad : {kNaN, kInf, -kInf}) {
        CHECK(throws_as<std::invalid_argument>([&] { h.add(3, std::vector<float>{bad, 0}); }));
        CHECK(throws_as<std::invalid_argument>([&] { h.search(std::vector<float>{0, bad}, 1); }));
    }
    CHECK(throws_as<std::invalid_argument>([&] { h.add(3, std::vector<float>{0, 0, 0}); }));
    CHECK(throws_as<std::invalid_argument>([&] { h.add(1, std::vector<float>{5, 5}); }));
    CHECK(throws_as<std::invalid_argument>([&] { h.search(std::vector<float>{0}, 1); }));
    CHECK(h.size() == 2 && h.storage().size() == 2);  // nothing half-inserted
    CHECK(h.entry_point() == entry && h.max_level() == top);
    passed("HnswIndex: invalid params and input rejected, index unchanged");
}

static void test_hnsw_failed_inserts_do_not_change_graph() {
    // Rejected inserts must not consume random numbers, or later levels would differ.
    auto data = clustered(300, 8, 5, 21);
    HnswIndex clean(8, Metric::L2), noisy(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        clean.add(i, data[i]);
        noisy.add(i, data[i]);
        if (i > 0 && i % 10 == 0) {
            CHECK(throws([&] { noisy.add(0, data[i]); }));                      // duplicate
            CHECK(throws([&] { noisy.add(9999, std::vector<float>(8, kNaN)); }));  // NaN
        }
    }
    CHECK(clean.entry_point() == noisy.entry_point() && clean.max_level() == noisy.max_level());
    for (std::size_t i = 0; i < 30; ++i) {
        auto a = clean.search(data[i], 10), b = noisy.search(data[i], 10);
        CHECK(a.size() == b.size());
        for (std::size_t j = 0; j < a.size(); ++j) CHECK(a[j].id == b[j].id && a[j].distance == b[j].distance);
    }
    passed("HnswIndex: rejected inserts leave later graph identical");
}

static void test_hnsw_tiny_and_extreme_sizes() {
    // 1, 2 and 3 vectors: results must exactly match the flat index.
    for (std::size_t n = 1; n <= 3; ++n) {
        HnswIndex h(4, Metric::L2);
        FlatIndex f(4, Metric::L2);
        auto data = random_vectors(n, 4, unsigned(30 + n));
        for (std::size_t i = 0; i < n; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        auto q = random_vectors(1, 4, 99)[0];
        auto hr = h.search(q, 10), fr = f.search(q, 10);
        CHECK(hr.size() == n && fr.size() == n);
        for (std::size_t i = 0; i < n; ++i) CHECK(hr[i].id == fr[i].id && hr[i].distance == fr[i].distance);
        check_graph(h, 1.0);
    }

    // dim = 1 on a number line.
    HnswIndex line(1, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) line.add(i, std::vector<float>{float(i)});
    auto r = line.search(std::vector<float>{50.2f}, 3, 50);
    CHECK(r.size() == 3 && r[0].id == 50 && r[1].id == 51 && r[2].id == 49);

    // Huge k and ef, and ef = 0, on a small connected index.
    HnswIndex small(4, Metric::L2);
    auto data = random_vectors(50, 4, 40);
    for (std::size_t i = 0; i < data.size(); ++i) small.add(i, data[i]);
    CHECK(small.search(data[0], kHuge, kHuge).size() == 50);
    CHECK(small.search(data[0], 5, 0).size() == 5);  // ef raised to k
    check_results(small, small.search(data[0], kHuge, kHuge));
    passed("HnswIndex: 1-3 vectors match flat, dim 1, huge k and ef, ef = 0");
}

static void test_hnsw_parameter_extremes() {
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    FlatIndex flat(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) flat.add(i, data[i]);

    // Extreme settings still build a valid graph, but quality drops as expected.
    // Measured: M=2 reaches 89.5% of nodes with recall 0.74; ef_construction=1
    // gives recall 0.66. Thresholds leave a safety margin below those values.
    struct Case { HnswParams params; double min_recall; double min_reachable; };
    const Case cases[] = {
        {{2, 200, 1}, 0.60, 0.80},     // smallest M: very sparse graph
        {{16, 1, 1}, 0.50, 0.99},      // smallest ef_construction: greedy build
        {{64, 200, 1}, 0.99, 0.999},   // large M
        {{16, 200, 777}, 0.95, 0.99},  // different seed
    };
    for (const Case& c : cases) {
        HnswIndex h(16, Metric::L2, c.params);
        for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
        check_graph(h, c.min_reachable);
        CHECK(recall(h, flat, queries, 10, 100) >= c.min_recall);
        for (const auto& q : queries) check_results(h, h.search(q, 10, 100));
    }
    passed("HnswIndex: M = 2, ef_construction = 1, M = 64, other seed");
}

static void test_hnsw_duplicates_and_metrics() {
    // Many identical vectors among random ones. Each node has limited link slots,
    // so not every copy can stay reachable (deduplicate data if that matters),
    // but the copies must not damage the rest of the graph.
    HnswIndex dup(8, Metric::L2);
    std::vector<float> same(8, 0.5f);
    auto noise = random_vectors(200, 8, 60);
    for (std::uint64_t i = 0; i < 200; ++i) {
        dup.add(i, same);             // IDs 0..199: identical
        dup.add(1000 + i, noise[i]);  // IDs 1000..1199: random
    }
    auto r = dup.search(same, 10, 100);
    CHECK(r.size() == 10);
    for (const auto& x : r) CHECK(x.id < 200 && x.distance == 0.0f);  // copies are found
    check_results(dup, r);
    check_graph(dup, 0.70);
    for (std::uint64_t i = 0; i < 200; ++i) {  // every unrelated vector still finds itself
        auto self = dup.search(noise[i], 1, 100);
        CHECK(self.size() == 1 && self[0].id == 1000 + i && self[0].distance == 0.0f);
    }

    // Inner product ordering.
    HnswIndex ip(2, Metric::InnerProduct);
    for (std::uint64_t i = 1; i <= 20; ++i) ip.add(i, std::vector<float>{float(i), 0});
    r = ip.search(std::vector<float>{1, 0}, 3, 50);
    CHECK(r[0].id == 20 && r[1].id == 19 && r[2].id == 18);

    // Cosine ignores length: a scaled copy is at distance ~0.
    HnswIndex cos(4, Metric::Cosine);
    auto data = random_vectors(100, 4, 61);
    for (std::size_t i = 0; i < data.size(); ++i) cos.add(i, data[i]);
    std::vector<float> scaled = data[7];
    for (float& x : scaled) x *= 1000.0f;
    r = cos.search(scaled, 1, 50);
    CHECK(r[0].id == 7 && std::fabs(r[0].distance) < 1e-5f);
    passed("HnswIndex: 200 identical vectors harmless, inner product order, cosine scale");
}

static void test_hnsw_removal_extremes() {
    auto data = clustered(400, 8, 4, 70);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }

    // Distances reported by HNSW are exactly the flat distances.
    for (std::size_t i = 0; i < 20; ++i) {
        std::vector<SearchResult> fr = f.search(data[i], 400);
        for (const auto& x : h.search(data[i], 10, 100)) {
            auto it = std::find_if(fr.begin(), fr.end(), [&](const SearchResult& y) { return y.id == x.id; });
            CHECK(it != fr.end() && it->distance == x.distance);
        }
    }

    // A removed vector is never returned, even when searching for it exactly.
    CHECK(h.remove(5));
    for (const auto& x : h.search(data[5], 10, 100)) CHECK(x.id != 5);

    // Remove all but one (including the entry point): that one is still found.
    for (std::uint64_t i = 0; i < 400; ++i)
        if (i != 123) h.remove(i);
    CHECK(h.size() == 1);
    auto r = h.search(data[0], 10, 400);
    CHECK(r.size() == 1 && r[0].id == 123);

    // Remove the last one, then insert new vectors into the all-removed graph.
    CHECK(h.remove(123) && h.size() == 0 && h.search(data[0], 5).empty());
    h.add(5000, data[0]);
    h.add(5001, data[1]);
    r = h.search(data[0], 5, 100);
    CHECK(r.size() == 2 && r[0].id == 5000);
    check_graph(h, 0.95);
    passed("HnswIndex: exact distances, remove all but one, refill after removing all");
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

static void test_concurrent_searches() {
    auto data = clustered(1500, 16, 10, 80);
    auto queries = clustered(100, 16, 10, 81);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }

    // Expected results, computed on one thread.
    std::vector<std::vector<SearchResult>> expect_h, expect_f;
    for (const auto& q : queries) {
        expect_h.push_back(h.search(q, 10, 64));
        expect_f.push_back(f.search(q, 10));
    }
    auto same = [](const std::vector<SearchResult>& a, const std::vector<SearchResult>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].id != b[i].id || a[i].distance != b[i].distance) return false;
        return true;
    };

    // 8 threads search both indexes at the same time; every answer must match.
    std::atomic<int> mismatches{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < 3; ++round)
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    std::size_t q = (i + std::size_t(t) * 13) % queries.size();
                    if (!same(h.search(queries[q], 10, 64), expect_h[q])) ++mismatches;
                    if (!same(f.search(queries[q], 10), expect_f[q])) ++mismatches;
                }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(mismatches.load() == 0);
    passed("Concurrency: 8 threads searching HNSW and Flat at once");
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

static void test_end_to_end_lifecycle() {
    for (Metric m : kMetrics) {
        // Inner product on un-normalized data is harder for HNSW: it is not a true
        // distance (a vector need not be closest to itself). On this data the
        // reference hnswlib reaches 0.887 / 0.900 / 0.853 recall at the three stages
        // below, and this index 0.898 / 0.905 / 0.860, so inner product gets a
        // lower threshold than L2 and cosine.
        const double min_recall = (m == Metric::InnerProduct) ? 0.80 : 0.95;
        const std::size_t dim = 24;
        auto data = clustered(1600, dim, 12, 90);
        auto queries = clustered(60, dim, 12, 91);
        HnswIndex h(dim, m);
        FlatIndex f(dim, m);

        // 1. Add 1000 vectors.
        for (std::size_t i = 0; i < 1000; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        CHECK(h.size() == 1000 && f.size() == 1000);
        CHECK(recall(h, f, queries, 10, 100) >= min_recall);

        // 2. Remove 300 of them.
        for (std::size_t i = 0; i < 1000; i += 3)
            if (h.size() > 700) {
                CHECK(h.remove(i) && f.remove(i));
            }
        CHECK(h.size() == 700 && f.size() == 700);
        CHECK(recall(h, f, queries, 10, 100) >= min_recall);
        for (const auto& q : queries) check_results(h, h.search(q, 10, 100));

        // 3. Add 600 more.
        for (std::size_t i = 1000; i < 1600; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        CHECK(h.size() == 1300 && f.size() == 1300);
        CHECK(recall(h, f, queries, 10, 100) >= min_recall);
        for (const auto& q : queries) {
            check_results(h, h.search(q, 10, 100));
            check_results(f, f.search(q, 10));
        }
        // With inner product, short vectors are rarely anyone's best match, so a
        // few lose all links over time (measured: 98.5% of live nodes reachable).
        check_graph(h, m == Metric::InnerProduct ? 0.97 : 0.99);
    }
    passed("End to end: add, remove, add again, for L2, IP and cosine");
}

int main() {
    std::printf("Layer 1\n");
    test_round_up_and_constants();
    test_aligned_block();
    test_arena();
    test_vector_store();
    test_id_map();
    test_graph_storage();
    test_storage();

    std::printf("Layer 2 (active kernel: %s)\n", isa_name(active_isa()));
    test_kernel_edge_values();
    test_normalize();
    test_dispatch();

    std::printf("Layer 3\n");
    test_topk();
    test_prepared_vector();
    test_visited();
    test_flat_index();
    test_hnsw_params_and_input();
    test_hnsw_failed_inserts_do_not_change_graph();
    test_hnsw_tiny_and_extreme_sizes();
    test_hnsw_parameter_extremes();
    test_hnsw_duplicates_and_metrics();
    test_hnsw_removal_extremes();

    std::printf("Concurrency and end to end\n");
    test_concurrent_searches();
    test_end_to_end_lifecycle();

    std::printf("All comprehensive tests passed (%ld checks).\n", g_check_count);
}