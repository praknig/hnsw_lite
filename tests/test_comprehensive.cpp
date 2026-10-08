/**
 * @file test_comprehensive.cpp
 * @brief Every test scenario for hnsw-lite in one program, with a built-in runner.
 *
 * Contains all scenarios from the per-layer test files plus many more edge
 * cases, negative scenarios, stress and end-to-end tests. Each test has a
 * group and a name, and can be run on its own.
 *
 * Usage:
 *   test_comprehensive                      run every test
 *   test_comprehensive --list               list all tests
 *   test_comprehensive --group hnsw         run one group (repeatable)
 *   test_comprehensive recall               run tests whose "group.name" contains "recall"
 *   test_comprehensive flat.k_zero          run a single test
 *   test_comprehensive --exclude e2e        skip matching tests (repeatable)
 *   test_comprehensive --fail-fast          stop at the first failing test
 *   test_comprehensive --help               show this help
 *
 * Groups: layer1, layer2, helpers, flat, hnsw, robustness, deletion, concurrency,
 *         metadata, filter, planner, payload, batch, range, search_e2e, update, stress, e2e
 *
 * How it works:
 *  - TEST(group, name) defines a test and registers it before main() runs.
 *  - CHECK(cond) records a failure and continues; REQUIRE(cond) records a
 *    failure and stops the current test. Both work from any thread.
 *  - CHECK_THROWS_AS(Type, statement) checks that the statement throws
 *    exactly that exception type (or a type derived from it).
 *  - SKIP(reason) ends a test early as skipped (for example, out-of-memory
 *    tests under AddressSanitizer, which installs its own allocator).
 *  - Out-of-memory tests replace the global allocator with one that can be
 *    told to fail the Nth allocation, then try every N through an operation.
 *  - Each test reports PASS or FAIL with its time; a summary lists failures.
 *  - The exit code is 0 only if every selected test passed.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "arena.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "storage.h"
#include "thread_pool.h"

using namespace vecdb;

// ===========================================================================
// Allocation-failure injection (for the out-of-memory tests)
// ===========================================================================
//
// This program replaces the global operator new/delete. Normally they behave
// like the standard ones. After alloc_hook::fail_after(n), the allocation that
// comes after n more successful ones throws std::bad_alloc, exactly once.
// Sanitizers install their own allocator, so the hook is disabled under them.
// Define HNSW_TEST_NO_ALLOC_HOOK to disable it manually.

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || defined(HNSW_TEST_NO_ALLOC_HOOK)
#define HNSW_ALLOC_HOOK 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define HNSW_ALLOC_HOOK 0
#endif
#endif
#ifndef HNSW_ALLOC_HOOK
#define HNSW_ALLOC_HOOK 1
#endif

namespace alloc_hook {

std::atomic<long> countdown{-1};  ///< -1: never fail; n >= 0: fail after n more allocations
std::atomic<bool> fired{false};   ///< set when an injected failure was thrown

/// Arms the hook: the allocation after `n` more successful ones will fail.
void fail_after(long n) {
    fired = false;
    countdown = n;
}

/// Disarms the hook.
void disarm() { countdown = -1; }

/// Called by every allocation; throws std::bad_alloc when the countdown hits 0.
inline void maybe_fail() {
    const long c = countdown.load(std::memory_order_relaxed);
    if (c < 0) return;
    if (c == 0) {
        countdown = -1;  // fail exactly once, so cleanup code can still allocate
        fired = true;
        throw std::bad_alloc();
    }
    countdown = c - 1;
}

}  // namespace alloc_hook

#if HNSW_ALLOC_HOOK
#include <cstdlib>
#include <new>
#ifdef _WIN32
#include <malloc.h>
#endif

static void* hook_alloc(std::size_t n) {
    alloc_hook::maybe_fail();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}

static void* hook_alloc_aligned(std::size_t n, std::size_t align) {
    alloc_hook::maybe_fail();
    if (n == 0) n = 1;
#ifdef _WIN32
    void* p = _aligned_malloc(n, align);
#else
    void* p = std::aligned_alloc(align, (n + align - 1) / align * align);
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

static void hook_free_aligned(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void* operator new(std::size_t n) { return hook_alloc(n); }
void* operator new[](std::size_t n) { return hook_alloc(n); }
void* operator new(std::size_t n, std::align_val_t a) { return hook_alloc_aligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return hook_alloc_aligned(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { hook_free_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { hook_free_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { hook_free_aligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { hook_free_aligned(p); }
#endif

// ===========================================================================
// Test runner
// ===========================================================================

namespace {

/// One registered test.
struct TestCase {
    std::string group;
    std::string name;
    void (*fn)();
    std::string full_name() const { return group + "." + name; }
};

/// All registered tests, in the order they appear in this file.
std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

/// Adds a test to the registry. Called by TEST() before main() runs.
bool register_test(const char* group, const char* name, void (*fn)()) {
    registry().push_back({group, name, fn});
    return true;
}

std::atomic<long> g_checks{0};   ///< checks executed in the whole run
std::atomic<int> g_failures{0};  ///< failed checks in the current test

/// Thrown by REQUIRE to stop the current test.
struct RequireFailed {};

/// Thrown by SKIP to end the current test as skipped.
struct TestSkipped {
    std::string reason;
};

/// Records one check result; prints the first few failures of a test.
void report(bool ok, int line, const char* expr) {
    ++g_checks;
    if (ok) return;
    const int n = ++g_failures;
    if (n <= 5) std::printf("        FAILED line %d: %s\n", line, expr);
    if (n == 6) std::printf("        (more failures in this test not shown)\n");
}

}  // namespace

/// Defines and registers a test function.
#define TEST(group, name)                                                    \
    static void test_##group##_##name();                                     \
    [[maybe_unused]] static const bool registered_##group##_##name =         \
        register_test(#group, #name, &test_##group##_##name);                \
    static void test_##group##_##name()

/// Ends the current test as skipped, with a reason.
#define SKIP(reason) throw TestSkipped{reason}

/// Records a failure if `cond` is false and continues the test.
#define CHECK(cond) report(static_cast<bool>(cond), __LINE__, #cond)

/// Records a failure if `cond` is false and stops the test.
#define REQUIRE(cond)                                                        \
    do {                                                                     \
        const bool ok_ = static_cast<bool>(cond);                            \
        report(ok_, __LINE__, #cond);                                        \
        if (!ok_) throw RequireFailed{};                                     \
    } while (0)

/// Checks that the statement throws exception type E (or a derived type).
#define CHECK_THROWS_AS(E, ...)                                              \
    do {                                                                     \
        bool ok_ = false;                                                    \
        try {                                                                \
            __VA_ARGS__;                                                     \
        } catch (const E&) {                                                 \
            ok_ = true;                                                      \
        } catch (...) {                                                      \
        }                                                                    \
        report(ok_, __LINE__, "throws " #E ": " #__VA_ARGS__);               \
    } while (0)

/// Checks that the statement does not throw.
#define CHECK_NOTHROW(...)                                                   \
    do {                                                                     \
        bool ok_ = true;                                                     \
        try {                                                                \
            __VA_ARGS__;                                                     \
        } catch (...) {                                                      \
            ok_ = false;                                                     \
        }                                                                    \
        report(ok_, __LINE__, "no throw: " #__VA_ARGS__);                    \
    } while (0)

// ===========================================================================
// Shared helpers
// ===========================================================================

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr std::size_t kHuge = std::numeric_limits<std::size_t>::max();
constexpr std::uint64_t kMaxId = std::numeric_limits<std::uint64_t>::max();
using Vectors = std::vector<std::vector<float>>;
using Results = std::vector<SearchResult>;

const Isa kIsas[] = {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon};
const Metric kMetrics[] = {Metric::L2, Metric::InnerProduct, Metric::Cosine};

/// True if `p` is 64-byte aligned.
bool aligned(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kAlign == 0; }

/// Uniform random vectors in [-1, 1].
Vectors random_vectors(std::size_t count, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Vectors out(count, std::vector<float>(dim));
    for (auto& v : out)
        for (float& x : v) x = dist(rng);
    return out;
}

/// Points scattered around random centers (behaves like real embeddings).
Vectors clustered(std::size_t count, std::size_t dim, std::size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    Vectors centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (float& x : c) x = center(rng);
    Vectors out(count, std::vector<float>(dim));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t d = 0; d < dim; ++d) out[i][d] = centers[rng() % clusters][d] + noise(rng);
    return out;
}

/// True if two result lists have the same IDs and distances in the same order.
bool same_results(const Results& a, const Results& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].distance != b[i].distance) return false;
    return true;
}

/// Average recall@k of `hnsw` against the exact `flat` results.
double recall(const HnswIndex& hnsw, const FlatIndex& flat, const Vectors& queries,
              std::size_t k, std::size_t ef) {
    double total = 0.0;
    std::size_t counted = 0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : flat.search(q, k)) truth.insert(r.id);
        if (truth.empty()) continue;
        std::size_t hits = 0;
        for (const auto& r : hnsw.search(q, k, ef)) hits += truth.count(r.id);
        total += double(hits) / double(truth.size());
        ++counted;
    }
    return counted ? total / double(counted) : 1.0;
}

/// Checks that results are sorted, have unique IDs and only contain live IDs.
template <class Index>
void check_results(const Index& index, const Results& results) {
    std::set<std::uint64_t> seen;
    bool unique = true, live = true, sorted = true;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (!seen.insert(results[i].id).second) unique = false;
        if (!index.contains(results[i].id)) live = false;
        if (i > 0 && results[i - 1].distance > results[i].distance) sorted = false;
    }
    CHECK(unique);
    CHECK(live);
    CHECK(sorted);
}

/// Checks every structural rule of an HNSW graph, and that at least
/// `min_reachable` of the live nodes can be reached from the entry point.
/// Removed nodes are skipped. Stale links (to a node whose level is now below
/// the link's level, left behind by slot reuse; searches skip them) are allowed
/// but must be rare: at most `max_stale` of all links (2% by default).
void check_graph(const HnswIndex& index, double min_reachable, double max_stale = 0.02) {
    const Storage& st = index.storage();
    const GraphStorage& g = st.graph();
    const IdMap& ids = st.ids();
    const std::size_t n = st.size();
    std::size_t live = 0;
    for (NodeId node = 0; node < n; ++node) live += !ids.is_deleted(node);
    CHECK(live == index.size());
    if (live == 0) {
        CHECK(index.entry_point() == kEmpty && index.max_level() == -1);
        return;
    }
    REQUIRE(index.entry_point() < n);
    CHECK(!ids.is_deleted(index.entry_point()));
    CHECK(g.level(index.entry_point()) == index.max_level());
    bool structure_ok = true;
    std::size_t links = 0, stale = 0;
    for (NodeId node = 0; node < n; ++node) {
        if (ids.is_deleted(node)) continue;
        if (g.level(node) > index.max_level()) structure_ok = false;
        for (int level = 0; level <= g.level(node); ++level) {
            auto slots = g.links(node, level);
            const std::size_t count = GraphStorage::count(slots);
            if (count > (level == 0 ? g.M0() : g.M())) structure_ok = false;
            std::set<NodeId> seen;
            for (std::size_t i = 0; i < count; ++i) {
                const NodeId nb = slots[i];
                ++links;
                if (nb >= n || nb == node || !seen.insert(nb).second) structure_ok = false;
                else if (g.level(nb) < level) ++stale;
            }
            for (std::size_t i = count; i < slots.size(); ++i)
                if (slots[i] != kEmpty) structure_ok = false;
        }
    }
    CHECK(structure_ok);
    CHECK(double(stale) <= max_stale * double(links));

    // Reachability of LIVE nodes on level 0 (the walk may pass removed nodes).
    std::vector<bool> reached(n, false);
    std::vector<NodeId> stack{index.entry_point()};
    reached[index.entry_point()] = true;
    while (!stack.empty()) {
        const NodeId node = stack.back();
        stack.pop_back();
        for (NodeId nb : g.links(node, 0)) {
            if (nb == kEmpty) break;
            if (!reached[nb]) {
                reached[nb] = true;
                stack.push_back(nb);
            }
        }
    }
    std::size_t live_reached = 0;
    for (NodeId node = 0; node < n; ++node)
        if (!ids.is_deleted(node)) live_reached += reached[node];
    CHECK(double(live_reached) / double(live) >= min_reachable);
}

/// Results sorted by distance, then user ID: lets two result lists be compared
/// when equal distances may come back in a different order.
Results tie_sorted(Results r) {
    std::sort(r.begin(), r.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    return r;
}

/// Exact reference search: every distance (same kernel), sorted, first k.
/// User IDs are `id_offset + position in data`.
Results reference_search(const Vectors& data, const std::vector<bool>& removed,
                         const std::vector<float>& query, Metric m, std::size_t k,
                         std::uint64_t id_offset) {
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
    Results out;
    for (std::size_t i = 0; i < std::min(k, all.size()); ++i)
        out.push_back({id_offset + all[i].id, all[i].distance});
    return out;
}

/// Double-precision reference distance, plus the size of the summed terms
/// (which sets the acceptable float rounding error).
double reference_distance(Metric m, const float* a, const float* b, std::size_t n,
                          double& magnitude) {
    double sum = 0.0;
    magnitude = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double term = (m == Metric::L2) ? (double(a[i]) - b[i]) * (double(a[i]) - b[i])
                                              : double(a[i]) * b[i];
        sum += term;
        magnitude += std::fabs(term);
    }
    if (m == Metric::InnerProduct) return -sum;
    if (m == Metric::Cosine) return 1.0 - sum;
    return sum;
}

/// Adds the same data to both indexes; user IDs are `first_id + i`.
void fill(HnswIndex& h, FlatIndex& f, const Vectors& data, std::uint64_t first_id = 0) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(first_id + i, data[i]);
        f.add(first_id + i, data[i]);
    }
}

/// A shared HNSW + Flat pair on clustered data (3000 x 32), one per metric.
/// Built the first time a test needs it, then reused, which keeps the suite fast.
struct Fixture {
    Vectors data, queries;
    std::unique_ptr<HnswIndex> hnsw;
    std::unique_ptr<FlatIndex> flat;
};

const Fixture& fixture(Metric m) {
    static std::unique_ptr<Fixture> cache[3];
    auto& f = cache[static_cast<int>(m)];
    if (!f) {
        f = std::make_unique<Fixture>();
        f->data = clustered(3000, 32, 20, 1);
        f->queries = clustered(100, 32, 20, 2);
        f->hnsw = std::make_unique<HnswIndex>(32, m);
        f->flat = std::make_unique<FlatIndex>(32, m);
        fill(*f->hnsw, *f->flat, f->data);
    }
    return *f;
}

}  // namespace

// ===========================================================================
// Layer 1: common.h
// ===========================================================================

TEST(layer1, round_up_boundaries) {
    CHECK(round_up(0, 16) == 0);
    CHECK(round_up(1, 16) == 16);
    CHECK(round_up(15, 16) == 16);
    CHECK(round_up(16, 16) == 16);
    CHECK(round_up(17, 16) == 32);
    CHECK(round_up(100, 16) == 112);
    CHECK(round_up(64, 64) == 64);
    CHECK(round_up(65, 64) == 128);
    CHECK(round_up(7, 1) == 7);
}

TEST(layer1, constants) {
    CHECK(kAlign == 64);
    CHECK(kFloatsPerLine == 16);
    CHECK(kEmpty == std::numeric_limits<std::uint32_t>::max());
    CHECK(sizeof(NodeId) == 4);
}

// ===========================================================================
// Layer 1: AlignedBlock
// ===========================================================================

TEST(layer1, aligned_block_alignment_and_zero_fill) {
    AlignedBlock b(100);
    CHECK(aligned(b.data()));
    CHECK(b.size() == 128);
    bool zero = true;
    for (std::size_t i = 0; i < b.size(); ++i) zero = zero && b.data()[i] == std::byte{0};
    CHECK(zero);
}

TEST(layer1, aligned_block_size_rounding) {
    CHECK(AlignedBlock(1).size() == 64);
    CHECK(AlignedBlock(64).size() == 64);
    CHECK(AlignedBlock(65).size() == 128);
    CHECK(AlignedBlock(1000).size() == 1024);
}

TEST(layer1, aligned_block_zero_size) {
    AlignedBlock empty(0);
    CHECK(empty.size() == 0);
}

TEST(layer1, aligned_block_custom_fill) {
    AlignedBlock ff(10, 0xFF);
    CHECK(ff.size() == 64 && aligned(ff.data()));
    bool filled = true;
    for (std::size_t i = 0; i < ff.size(); ++i) filled = filled && ff.data()[i] == std::byte{0xFF};
    CHECK(filled);
}

TEST(layer1, aligned_block_move_construct) {
    AlignedBlock a(128);
    std::byte* original = a.data();
    AlignedBlock b = std::move(a);
    CHECK(a.data() == nullptr && a.size() == 0);
    CHECK(b.data() == original && b.size() == 128);
}

TEST(layer1, aligned_block_move_assign) {
    AlignedBlock a(128);
    std::memset(a.data(), 7, 128);
    AlignedBlock b(64);
    b = std::move(a);
    CHECK(a.data() == nullptr && a.size() == 0);
    CHECK(b.size() == 128 && b.data()[127] == std::byte{7});
}

TEST(layer1, aligned_block_self_move) {
    AlignedBlock b(128);
    std::memset(b.data(), 7, 128);
    AlignedBlock& alias = b;
    b = std::move(alias);
    CHECK(b.size() == 128 && b.data()[0] == std::byte{7});
}

// ===========================================================================
// Layer 1: Arena
// ===========================================================================

TEST(layer1, arena_starts_empty) {
    Arena a;
    CHECK(a.block_count() == 0);
}

TEST(layer1, arena_no_overlap) {
    Arena a(1024);
    auto* x = a.allocate_array<std::uint64_t>(10);
    auto* y = a.allocate_array<std::uint64_t>(10);
    CHECK(y >= x + 10);
}

TEST(layer1, arena_all_alignments) {
    Arena a(256);
    for (std::size_t align : std::initializer_list<std::size_t>{1, 2, 4, 8, 16, 32, 64}) {
        void* p = a.allocate(3, align);
        CHECK(reinterpret_cast<std::uintptr_t>(p) % align == 0);
    }
}

TEST(layer1, arena_exact_block_fill) {
    Arena a(256);
    a.allocate(256, 64);
    CHECK(a.block_count() == 1);
    a.allocate(1);
    CHECK(a.block_count() == 2);
}

TEST(layer1, arena_zero_bytes) {
    Arena a(64);
    CHECK(a.allocate(0) != nullptr);
}

TEST(layer1, arena_oversized_request) {
    Arena a(1024);
    a.allocate(10);
    a.allocate(5000);  // bigger than a block: gets its own block
    CHECK(a.block_count() == 2);
    void* small = a.allocate(8);
    CHECK(small != nullptr);
}

TEST(layer1, arena_many_blocks_stable) {
    Arena a(1024);
    std::vector<int*> arrays;
    for (int i = 0; i < 100; ++i) {
        int* p = a.allocate_array<int>(50);
        std::fill_n(p, 50, i);
        arrays.push_back(p);
    }
    CHECK(a.block_count() > 1);
    bool intact = true;
    for (int i = 0; i < 100; ++i)
        for (std::size_t j = 0; j < 50; ++j) intact = intact && arrays[std::size_t(i)][j] == i;
    CHECK(intact);
}

TEST(layer1, arena_typed_alignment) {
    Arena a(512);
    a.allocate(1);  // misalign the bookmark first
    auto* d = a.allocate_array<double>(3);
    auto* u = a.allocate_array<std::uint64_t>(3);
    CHECK(reinterpret_cast<std::uintptr_t>(d) % alignof(double) == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(u) % alignof(std::uint64_t) == 0);
}

// ===========================================================================
// Layer 1: VectorStore
// ===========================================================================

TEST(layer1, vector_store_strides) {
    const std::size_t dims[][2] = {{1, 16}, {16, 16}, {17, 32}, {31, 32}, {32, 32},
                                   {33, 48}, {100, 112}, {128, 128}, {768, 768}};
    for (const auto& d : dims) CHECK(VectorStore(d[0]).stride() == d[1]);
}

TEST(layer1, vector_store_rejects_zero_dim) {
    CHECK_THROWS_AS(std::invalid_argument, VectorStore bad(0));
}

TEST(layer1, vector_store_roundtrip) {
    VectorStore s(100);
    std::vector<float> v(100);
    std::iota(v.begin(), v.end(), 1.0f);
    NodeId id = s.add(v);
    auto got = s.get(id);
    CHECK(got.size() == 100);
    CHECK(std::equal(got.begin(), got.end(), v.begin()));
}

TEST(layer1, vector_store_rows_aligned_and_padded) {
    VectorStore s(100, 2);
    std::vector<float> v(100, 5.0f);
    for (int i = 0; i < 11; ++i) s.add(v);
    bool ok = true;
    for (NodeId id = 0; id < s.size(); ++id) {
        ok = ok && aligned(s.get(id).data());
        auto padded = s.get_padded(id);
        for (std::size_t i = 100; i < padded.size(); ++i) ok = ok && padded[i] == 0.0f;
    }
    CHECK(ok);
}

TEST(layer1, vector_store_addresses_stable) {
    VectorStore s(8, 2);  // 4 vectors per shelf
    std::vector<float> v(8, 1.0f);
    const float* first = s.get(s.add(v)).data();
    for (int i = 0; i < 100; ++i) s.add(v);
    CHECK(s.get(0).data() == first);
}

TEST(layer1, vector_store_one_row_per_shelf) {
    VectorStore s(3, 0);
    for (int i = 0; i < 5; ++i) s.add(std::vector<float>{float(i), 0, 0});
    bool ok = true;
    for (NodeId i = 0; i < 5; ++i) ok = ok && s.get(i)[0] == float(i) && aligned(s.get(i).data());
    CHECK(ok);
}

TEST(layer1, vector_store_unknown_id_throws) {
    VectorStore s(3);
    CHECK_THROWS_AS(std::out_of_range, s.get(0));
    s.add(std::vector<float>{1, 2, 3});
    CHECK_THROWS_AS(std::out_of_range, s.get(1));
    CHECK_THROWS_AS(std::out_of_range, s.get_padded(1));
}

TEST(layer1, vector_store_wrong_dim_unchanged) {
    VectorStore s(4);
    CHECK_THROWS_AS(std::invalid_argument, s.add(std::vector<float>(3)));
    CHECK_THROWS_AS(std::invalid_argument, s.add(std::vector<float>(5)));
    CHECK_THROWS_AS(std::invalid_argument, s.add(std::vector<float>{}));
    CHECK(s.size() == 0);
}

TEST(layer1, vector_store_special_values) {
    VectorStore s(4);
    const float max = std::numeric_limits<float>::max();
    const float denorm = std::numeric_limits<float>::denorm_min();
    NodeId id = s.add(std::vector<float>{-0.0f, max, denorm, kNaN});
    auto v = s.get(id);
    CHECK(std::signbit(v[0]) && v[0] == 0.0f);
    CHECK(v[1] == max && v[2] == denorm && std::isnan(v[3]));
}

TEST(layer1, vector_store_span_lengths) {
    VectorStore s(5);
    NodeId id = s.add(std::vector<float>(5, 1.0f));
    CHECK(s.get(id).size() == 5);
    CHECK(s.get_padded(id).size() == 16);
    CHECK(s.dim() == 5 && s.stride() == 16 && s.size() == 1);
}

TEST(layer1, vector_store_shelf_size_bounded) {
    CHECK(VectorStore(4).rows_per_shelf() == 65536);     // 64 B rows: 4 MiB blocks
    CHECK(VectorStore(128).rows_per_shelf() == 16384);   // 512 B rows: 8 MiB blocks
    CHECK(VectorStore(1536).rows_per_shelf() == 1024);   // 6 KiB rows: 6 MiB blocks
    CHECK(VectorStore(3, 0).rows_per_shelf() == 1);      // explicit size still works
    CHECK(VectorStore(3, 5).rows_per_shelf() == 32);
}

TEST(layer1, vector_store_many_vectors) {
    VectorStore s(7, 4);  // 16 vectors per shelf, so 10,000 vectors use 625 shelves
    for (int i = 0; i < 10000; ++i) s.add(std::vector<float>(7, float(i)));
    bool ok = s.size() == 10000;
    for (NodeId i = 0; i < 10000; i += 37) ok = ok && s.get(i)[6] == float(i);
    CHECK(ok);
}

// ===========================================================================
// Layer 1: IdMap
// ===========================================================================

TEST(layer1, id_map_sequential_numbers) {
    IdMap m;
    CHECK(m.add(5000) == 0);
    CHECK(m.add(42) == 1);
    CHECK(m.add(7) == 2);
    CHECK(m.size() == 3);
    CHECK(m.external(0) == 5000 && m.external(1) == 42);
}

TEST(layer1, id_map_duplicate_rejected) {
    IdMap m;
    m.add(42);
    CHECK_THROWS_AS(std::invalid_argument, m.add(42));
    CHECK(m.size() == 1);
}

TEST(layer1, id_map_find) {
    IdMap m;
    m.add(42);
    CHECK(m.find(42).has_value() && *m.find(42) == 0);
    CHECK(!m.find(7).has_value());
}

TEST(layer1, id_map_extreme_ids) {
    IdMap m;
    CHECK(m.add(0) == 0 && m.add(kMaxId) == 1);
    CHECK(*m.find(0) == 0 && *m.find(kMaxId) == 1);
    CHECK(m.external(1) == kMaxId);
}

TEST(layer1, id_map_unknown_internal_throws) {
    IdMap m;
    CHECK_THROWS_AS(std::out_of_range, m.external(0));
    CHECK_THROWS_AS(std::out_of_range, m.mark_deleted(0));
    CHECK_THROWS_AS(std::out_of_range, m.is_deleted(0));
}

TEST(layer1, id_map_delete_flags) {
    IdMap m;
    m.add(1);
    m.add(2);
    m.mark_deleted(1);
    CHECK(m.is_deleted(1) && !m.is_deleted(0));
    m.mark_deleted(1);  // twice is harmless
    CHECK(m.is_deleted(1) && m.size() == 2);
}

TEST(layer1, id_map_removed_id_stays_taken) {
    IdMap m;
    m.add(9);
    m.mark_deleted(0);
    CHECK_THROWS_AS(std::invalid_argument, m.add(9));
    CHECK(m.find(9).has_value());
}

// ===========================================================================
// Layer 1: GraphStorage
// ===========================================================================

TEST(layer1, graph_rejects_zero_M) {
    CHECK_THROWS_AS(std::invalid_argument, GraphStorage bad(0));
}

TEST(layer1, graph_capacities) {
    GraphStorage g(16);
    NodeId a = g.add_node(2);
    CHECK(g.M() == 16 && g.M0() == 32);
    CHECK(g.links(a, 0).size() == 32);
    CHECK(g.links(a, 1).size() == 16 && g.links(a, 2).size() == 16);
}

TEST(layer1, graph_new_lists_empty) {
    GraphStorage g(16);
    NodeId a = g.add_node(3);
    bool empty = true;
    for (int l = 0; l <= 3; ++l) empty = empty && GraphStorage::count(g.links(a, l)) == 0;
    CHECK(empty);
}

TEST(layer1, graph_level_limits) {
    GraphStorage g(1);
    NodeId top = g.add_node(255);
    CHECK(g.level(top) == 255 && g.links(top, 255).size() == 1);
    CHECK_THROWS_AS(std::invalid_argument, g.add_node(256));
    CHECK_THROWS_AS(std::invalid_argument, g.add_node(-1));
    CHECK(g.size() == 1);
}

TEST(layer1, graph_wrong_level_throws) {
    GraphStorage g(16);
    NodeId a = g.add_node(0);
    CHECK_THROWS_AS(std::out_of_range, g.links(a, 1));
    CHECK_THROWS_AS(std::out_of_range, g.links(a, -1));
}

TEST(layer1, graph_unknown_node_throws) {
    GraphStorage g(16);
    CHECK_THROWS_AS(std::out_of_range, g.links(0, 0));
    CHECK_THROWS_AS(std::out_of_range, g.level(0));
}

TEST(layer1, graph_write_read_links) {
    GraphStorage g(16);
    NodeId a = g.add_node(0), b = g.add_node(2);
    g.links(a, 0)[0] = b;
    g.links(a, 0)[1] = 7;
    g.links(b, 2)[0] = a;
    CHECK(GraphStorage::count(g.links(a, 0)) == 2);
    CHECK(g.links(b, 2)[0] == a);
    CHECK(GraphStorage::count(g.links(b, 1)) == 0);  // other levels untouched
}

TEST(layer1, graph_shelf_spill) {
    GraphStorage g(16, 2);  // 4 nodes per shelf
    for (int i = 0; i < 10; ++i) g.add_node(0);
    bool ok = true;
    for (NodeId i = 0; i < 10; ++i) {
        ok = ok && aligned(g.links(i, 0).data());
        g.links(i, 0)[0] = i + 100;
    }
    for (NodeId i = 0; i < 10; ++i) ok = ok && g.links(i, 0)[0] == i + 100;
    CHECK(ok);
}

TEST(layer1, graph_set_links_replaces) {
    GraphStorage g(16);
    NodeId a = g.add_node(0);
    g.set_links(a, 0, std::vector<NodeId>{2, 3, 4});
    CHECK(GraphStorage::count(g.links(a, 0)) == 3);
    g.set_links(a, 0, std::vector<NodeId>{5});
    CHECK(GraphStorage::count(g.links(a, 0)) == 1 && g.links(a, 0)[0] == 5);
    CHECK(g.links(a, 0)[1] == kEmpty);
}

TEST(layer1, graph_set_links_exact_capacity) {
    GraphStorage g(1);
    NodeId a = g.add_node(0);
    g.set_links(a, 0, std::vector<NodeId>{0, 5});
    CHECK(GraphStorage::count(g.links(a, 0)) == 2);
}

TEST(layer1, graph_set_links_too_long_unchanged) {
    GraphStorage g(1);
    NodeId a = g.add_node(0);
    g.set_links(a, 0, std::vector<NodeId>{0, 5});
    CHECK_THROWS_AS(std::length_error, g.set_links(a, 0, std::vector<NodeId>{1, 2, 3}));
    CHECK(g.links(a, 0)[0] == 0 && g.links(a, 0)[1] == 5);
}

TEST(layer1, graph_set_links_empty_clears) {
    GraphStorage g(16);
    NodeId a = g.add_node(0);
    g.set_links(a, 0, std::vector<NodeId>{1, 2});
    g.set_links(a, 0, std::vector<NodeId>{});
    CHECK(GraphStorage::count(g.links(a, 0)) == 0);
}

TEST(layer1, graph_count_on_custom_spans) {
    std::vector<NodeId> none{kEmpty, kEmpty}, full{1, 2, 3}, part{1, kEmpty, 3};
    CHECK(GraphStorage::count(none) == 0);
    CHECK(GraphStorage::count(full) == 3);
    CHECK(GraphStorage::count(part) == 1);  // stops at the first kEmpty
}

TEST(layer1, graph_const_access) {
    GraphStorage g(16);
    NodeId a = g.add_node(1);
    const GraphStorage& cg = g;
    CHECK(cg.links(a, 0).size() == 32 && cg.links(a, 1).size() == 16);
    CHECK(cg.level(a) == 1 && cg.size() == 1);
}

// ===========================================================================
// Layer 1: Storage
// ===========================================================================

TEST(layer1, storage_insert_and_lookup) {
    Storage st(3);
    NodeId id = st.insert(777, std::vector<float>{1, 2, 3}, 1);
    CHECK(st.vectors().get(id)[2] == 3.0f);
    CHECK(st.graph().level(id) == 1);
    CHECK(*st.ids().find(777) == id);
    CHECK(st.size() == 1);
}

TEST(layer1, storage_rejects_wrong_dim) {
    Storage st(3);
    CHECK_THROWS_AS(std::invalid_argument, st.insert(1, std::vector<float>(2), 0));
    CHECK(st.size() == 0 && st.ids().size() == 0 && st.graph().size() == 0);
}

TEST(layer1, storage_rejects_bad_level) {
    Storage st(3);
    std::vector<float> v{1, 2, 3};
    CHECK_THROWS_AS(std::invalid_argument, st.insert(1, v, -1));
    CHECK_THROWS_AS(std::invalid_argument, st.insert(1, v, 256));
    CHECK(st.size() == 0 && st.ids().size() == 0 && st.graph().size() == 0);
}

TEST(layer1, storage_rejects_duplicate) {
    Storage st(3);
    std::vector<float> v{1, 2, 3};
    st.insert(777, v, 0);
    CHECK_THROWS_AS(std::invalid_argument, st.insert(777, v, 0));
    CHECK(st.size() == 1 && st.ids().size() == 1 && st.graph().size() == 1);
}

TEST(layer1, storage_stays_in_sync) {
    Storage st(2);
    std::vector<float> v{1, 2};
    for (std::uint64_t i = 0; i < 200; ++i) {
        st.insert(i, v, int(i % 4));
        if (i % 7 == 0) {
            try { st.insert(i, v, 0); } catch (...) {}                     // duplicate
            try { st.insert(1000 + i, v, -1); } catch (...) {}             // bad level
            try { st.insert(2000 + i, std::vector<float>(3), 0); } catch (...) {}  // bad dim
        }
    }
    CHECK(st.size() == 200);
    CHECK(st.ids().size() == 200 && st.graph().size() == 200);
    CHECK(st.graph().level(5) == 1 && st.ids().external(199) == 199);
}

TEST(layer1, storage_level_lists_created) {
    Storage st(2);
    NodeId id = st.insert(1, std::vector<float>{1, 2}, 3);
    bool ok = true;
    for (int l = 0; l <= 3; ++l) ok = ok && GraphStorage::count(st.graph().links(id, l)) == 0;
    CHECK(ok);
    CHECK_THROWS_AS(std::out_of_range, st.graph().links(id, 4));
}

TEST(layer1, storage_insert_into_validation) {
    // Found by the coverage audit: insert_into's own checks had no test.
    Storage st(4);
    std::vector<float> v{1, 2, 3, 4};
    st.insert(10, v, 0);
    st.insert(11, v, 0);
    st.ids().release(11);  // slot 1 is free
    CHECK_THROWS_AS(std::invalid_argument, st.insert_into(1, 20, std::vector<float>(3), 0));
    CHECK_THROWS_AS(std::invalid_argument, st.insert_into(1, 20, v, -1));
    CHECK_THROWS_AS(std::invalid_argument, st.insert_into(1, 20, v, 256));
    CHECK_THROWS_AS(std::invalid_argument, st.insert_into(1, 10, v, 0));  // ID in use
    CHECK_THROWS_AS(std::logic_error, st.insert_into(0, 20, v, 0));       // live slot
    CHECK_THROWS_AS(std::logic_error, st.insert_into(9, 20, v, 0));       // no such slot
    CHECK(st.ids().is_free(1) && !st.ids().find(20).has_value() && st.size() == 2);
    CHECK(st.insert_into(1, 20, std::vector<float>{5, 6, 7, 8}, 2) == 1);
    CHECK(st.vectors().get(1)[3] == 8.0f && st.graph().level(1) == 2);
}

TEST(layer1, vector_store_huge_dimension) {
    // Rows over 8 MiB: each block holds a single row (found by the coverage audit).
    VectorStore s(3000000);
    CHECK(s.rows_per_shelf() == 1);
    s.add(std::vector<float>(3000000, 1.5f));
    s.add(std::vector<float>(3000000, 2.5f));
    CHECK(s.get(0)[2999999] == 1.5f && s.get(1)[0] == 2.5f);
    CHECK(aligned(s.get(0).data()) && aligned(s.get(1).data()));
}

// ===========================================================================
// Layer 2: dispatch
// ===========================================================================

TEST(layer2, dispatch_scalar_always_supported) {
    CHECK(isa_supported(Isa::Scalar));
    for (Metric m : kMetrics) CHECK(get_distance(m, Isa::Scalar) != nullptr);
}

TEST(layer2, dispatch_active_is_stable_and_supported) {
    const Isa first = active_isa();
    CHECK(isa_supported(first));
    for (int i = 0; i < 10; ++i) CHECK(active_isa() == first);
}

TEST(layer2, dispatch_default_matches_active) {
    for (Metric m : kMetrics) CHECK(get_distance(m) == get_distance(m, active_isa()));
    for (Metric m : kMetrics) CHECK(get_distance(m) != nullptr);
}

TEST(layer2, dispatch_nullptr_iff_unsupported) {
    for (Isa isa : kIsas)
        for (Metric m : kMetrics) CHECK((get_distance(m, isa) != nullptr) == isa_supported(isa));
}

TEST(layer2, dispatch_never_neon_and_x86) {
    CHECK(!(isa_supported(Isa::Neon) && isa_supported(Isa::Avx2)));
    CHECK(!(isa_supported(Isa::Neon) && isa_supported(Isa::Avx512)));
}

TEST(layer2, dispatch_isa_names) {
    CHECK(std::strcmp(isa_name(Isa::Scalar), "scalar") == 0);
    CHECK(std::strcmp(isa_name(Isa::Avx2), "avx2") == 0);
    CHECK(std::strcmp(isa_name(Isa::Avx512), "avx512") == 0);
    CHECK(std::strcmp(isa_name(Isa::Neon), "neon") == 0);
}

TEST(layer2, dispatch_metrics_differ) {
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(get_distance(Metric::L2, isa) != get_distance(Metric::InnerProduct, isa));
        CHECK(get_distance(Metric::L2, isa) != get_distance(Metric::Cosine, isa));
        CHECK(get_distance(Metric::InnerProduct, isa) != get_distance(Metric::Cosine, isa));
    }
}

// ===========================================================================
// Layer 2: kernels
// ===========================================================================

TEST(layer2, kernels_match_reference) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (std::size_t n : std::initializer_list<std::size_t>{16, 32, 48, 64, 80, 112, 128, 768, 1536}) {
            std::vector<float> a(n), b(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = dist(rng), b[i] = dist(rng);
            for (Metric m : kMetrics) {
                double magnitude = 0.0;
                const double expected = reference_distance(m, a.data(), b.data(), n, magnitude);
                const float got = get_distance(m, isa)(a.data(), b.data(), n);
                CHECK(std::fabs(got - expected) <= 1e-5 * magnitude + 1e-6);
            }
        }
    }
}

TEST(layer2, kernels_known_values) {
    std::vector<float> a(16, 0.0f), b(16, 0.0f);
    a[0] = 1, a[1] = 2;  // [1, 2, 0, ...]
    b[0] = 4, b[1] = 6;  // [4, 6, 0, ...]
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(get_distance(Metric::L2, isa)(a.data(), b.data(), 16) == 25.0f);
        CHECK(get_distance(Metric::InnerProduct, isa)(a.data(), b.data(), 16) == -16.0f);
        CHECK(get_distance(Metric::Cosine, isa)(a.data(), b.data(), 16) == -15.0f);  // 1 - 16
    }
}

TEST(layer2, kernels_identical_and_orthogonal) {
    std::vector<float> x(16, 0.0f), y(16, 0.0f);
    x[0] = 1;
    y[1] = 1;
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(get_distance(Metric::L2, isa)(x.data(), x.data(), 16) == 0.0f);
        CHECK(get_distance(Metric::L2, isa)(x.data(), y.data(), 16) == 2.0f);
        CHECK(get_distance(Metric::Cosine, isa)(x.data(), x.data(), 16) == 0.0f);
        CHECK(get_distance(Metric::Cosine, isa)(x.data(), y.data(), 16) == 1.0f);
        CHECK(get_distance(Metric::InnerProduct, isa)(x.data(), y.data(), 16) == 0.0f);
    }
}

TEST(layer2, kernels_length_zero) {
    std::vector<float> z(16, 0.0f);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(get_distance(Metric::L2, isa)(z.data(), z.data(), 0) == 0.0f);
        CHECK(get_distance(Metric::InnerProduct, isa)(z.data(), z.data(), 0) == 0.0f);
        CHECK(get_distance(Metric::Cosine, isa)(z.data(), z.data(), 0) == 1.0f);
    }
}

TEST(layer2, kernels_zero_vectors) {
    std::vector<float> z(64, 0.0f);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(get_distance(Metric::L2, isa)(z.data(), z.data(), 64) == 0.0f);
        CHECK(get_distance(Metric::InnerProduct, isa)(z.data(), z.data(), 64) == 0.0f);
        CHECK(get_distance(Metric::Cosine, isa)(z.data(), z.data(), 64) == 1.0f);
    }
}

TEST(layer2, kernels_overflow_to_infinity) {
    std::vector<float> a(16, 0.0f), b(16, 0.0f);
    a[0] = 3e38f;
    b[0] = -3e38f;
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(std::isinf(get_distance(Metric::L2, isa)(a.data(), b.data(), 16)));
    }
}

TEST(layer2, kernels_nan_and_inf_propagate) {
    std::vector<float> a(32, 0.5f), b(32, 0.5f);
    a[17] = kNaN;
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (Metric m : kMetrics) CHECK(std::isnan(get_distance(m, isa)(a.data(), b.data(), 32)));
    }
    a[17] = kInf;
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        CHECK(std::isinf(get_distance(Metric::L2, isa)(a.data(), b.data(), 32)));
    }
}

TEST(layer2, kernels_symmetric) {
    auto r = random_vectors(2, 4096, 11);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (std::size_t n : std::initializer_list<std::size_t>{16, 48, 80, 4096})
            for (Metric m : kMetrics) {
                auto fn = get_distance(m, isa);
                CHECK(fn(r[0].data(), r[1].data(), n) == fn(r[1].data(), r[0].data(), n));
            }
    }
}

TEST(layer2, kernels_l2_non_negative_and_self_zero) {
    auto r = random_vectors(20, 128, 12);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        auto l2 = get_distance(Metric::L2, isa);
        for (std::size_t i = 0; i + 1 < r.size(); ++i) {
            CHECK(l2(r[i].data(), r[i + 1].data(), 128) >= 0.0f);
            CHECK(l2(r[i].data(), r[i].data(), 128) == 0.0f);
        }
    }
}

TEST(layer2, kernels_unaligned_input) {
    auto src = random_vectors(2, 64, 13);
    std::vector<float> shifted(65), copy(src[0]);
    std::copy(src[0].begin(), src[0].end(), shifted.begin() + 1);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (Metric m : kMetrics) {
            auto fn = get_distance(m, isa);
            CHECK(fn(shifted.data() + 1, src[1].data(), 64) == fn(copy.data(), src[1].data(), 64));
        }
    }
}

TEST(layer2, kernels_padding_harmless) {
    auto r = random_vectors(2, 100, 7);
    VectorStore store(100);
    NodeId ia = store.add(r[0]), ib = store.add(r[1]);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (Metric m : kMetrics) {
            double magnitude = 0.0;
            const double expected = reference_distance(m, r[0].data(), r[1].data(), 100, magnitude);
            const float got = get_distance(m, isa)(store.get_padded(ia).data(),
                                                   store.get_padded(ib).data(), store.stride());
            CHECK(std::fabs(got - expected) <= 1e-5 * magnitude + 1e-6);
        }
    }
}

TEST(layer2, kernels_long_vectors) {
    auto r = random_vectors(2, 8192, 14);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        for (Metric m : kMetrics) {
            double magnitude = 0.0;
            const double expected = reference_distance(m, r[0].data(), r[1].data(), 8192, magnitude);
            const float got = get_distance(m, isa)(r[0].data(), r[1].data(), 8192);
            CHECK(std::fabs(got - expected) <= 1e-5 * magnitude + 1e-6);
        }
    }
}

TEST(layer2, kernels_cosine_is_one_plus_ip) {
    // Both come from the same dot product: cos = 1 - dot, ip = -dot.
    auto r = random_vectors(2, 96, 15);
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        const float ip = get_distance(Metric::InnerProduct, isa)(r[0].data(), r[1].data(), 96);
        const float cos = get_distance(Metric::Cosine, isa)(r[0].data(), r[1].data(), 96);
        CHECK(cos == 1.0f + ip);
    }
}

TEST(layer2, kernels_l2_scales_by_four) {
    // Doubling both vectors is exact in floating point, so L2 is exactly 4x.
    auto r = random_vectors(2, 80, 16);
    std::vector<float> a2(r[0]), b2(r[1]);
    for (float& x : a2) x *= 2.0f;
    for (float& x : b2) x *= 2.0f;
    for (Isa isa : kIsas) {
        if (!isa_supported(isa)) continue;
        auto l2 = get_distance(Metric::L2, isa);
        CHECK(l2(a2.data(), b2.data(), 80) == 4.0f * l2(r[0].data(), r[1].data(), 80));
    }
}

TEST(layer2, kernels_versions_agree) {
    auto r = random_vectors(2, 1024, 17);
    for (Metric m : kMetrics) {
        const float base = get_distance(m, Isa::Scalar)(r[0].data(), r[1].data(), 1024);
        double magnitude = 0.0;
        reference_distance(m, r[0].data(), r[1].data(), 1024, magnitude);
        for (Isa isa : kIsas) {
            if (!isa_supported(isa)) continue;
            const float other = get_distance(m, isa)(r[0].data(), r[1].data(), 1024);
            CHECK(std::fabs(other - base) <= 2e-5 * magnitude + 1e-6);
        }
    }
}

// ===========================================================================
// Layer 2: normalize
// ===========================================================================

TEST(layer2, normalize_known_values) {
    std::vector<float> v{3.0f, 4.0f};
    normalize(v);
    CHECK(std::fabs(v[0] - 0.6f) < 1e-6f && std::fabs(v[1] - 0.8f) < 1e-6f);
}

TEST(layer2, normalize_unit_length_many_dims) {
    bool ok = true;
    for (std::size_t dim = 1; dim <= 1536; dim = dim * 2 + 1) {
        auto v = random_vectors(1, dim, unsigned(dim))[0];
        normalize(v);
        double len2 = 0.0;
        for (float x : v) len2 += double(x) * x;
        ok = ok && std::fabs(len2 - 1.0) < 1e-5;
    }
    CHECK(ok);
}

TEST(layer2, normalize_zero_unchanged) {
    std::vector<float> z(16, 0.0f);
    normalize(z);
    bool ok = true;
    for (float x : z) ok = ok && x == 0.0f;
    CHECK(ok);
}

TEST(layer2, normalize_empty) {
    std::vector<float> empty;
    CHECK_NOTHROW(normalize(empty));
}

TEST(layer2, normalize_negative) {
    std::vector<float> v{-5.0f};
    normalize(v);
    CHECK(v[0] == -1.0f);
}

TEST(layer2, normalize_tiny) {
    std::vector<float> v{1e-40f, 0.0f, 0.0f};
    normalize(v);
    CHECK(std::fabs(v[0] - 1.0f) < 1e-6f && v[1] == 0.0f && v[2] == 0.0f);
}

TEST(layer2, normalize_huge) {
    std::vector<float> v{1e30f, 1e30f};
    normalize(v);
    CHECK(std::fabs(v[0] - 0.70710678f) < 1e-6f && std::fabs(v[1] - 0.70710678f) < 1e-6f);
}

TEST(layer2, normalize_idempotent) {
    auto v = random_vectors(1, 50, 18)[0];
    normalize(v);
    auto once = v;
    normalize(v);
    bool ok = true;
    for (std::size_t i = 0; i < v.size(); ++i) ok = ok && std::fabs(v[i] - once[i]) < 1e-6f;
    CHECK(ok);
}

TEST(layer2, normalize_keeps_direction) {
    std::vector<float> v{2.0f, -4.0f, 8.0f};
    normalize(v);
    CHECK(v[0] > 0 && v[1] < 0 && v[2] > 0);
    CHECK(std::fabs(v[1] / v[0] + 2.0f) < 1e-5f && std::fabs(v[2] / v[0] - 4.0f) < 1e-5f);
}

// ===========================================================================
// Layer 3 helpers: Candidate and TopK
// ===========================================================================

TEST(helpers, candidate_ordering) {
    CHECK((Candidate{1.0f, 5} < Candidate{2.0f, 1}));   // distance first
    CHECK((Candidate{1.0f, 1} < Candidate{1.0f, 2}));   // then id
    CHECK(!(Candidate{1.0f, 2} < Candidate{1.0f, 2}));  // not less than itself
    CHECK((Candidate{2.0f, 0} > Candidate{1.0f, 9}));
    CHECK((Candidate{-1.0f, 9} < Candidate{0.0f, 0}));  // negative distances (inner product)
    CHECK(!(Candidate{1.0f, 5} < Candidate{1.0f, 3}));  // equal distance, larger id
}

TEST(helpers, topk_keeps_best_sorted) {
    TopK top(3);
    const float distances[] = {5, 1, 4, 2, 3, 0.5f, 9};
    for (NodeId i = 0; i < 7; ++i) top.push({distances[i], i});
    auto best = top.take_sorted();
    REQUIRE(best.size() == 3);
    CHECK(best[0].id == 5 && best[1].id == 1 && best[2].id == 3);  // 0.5, 1, 2
}

TEST(helpers, topk_state) {
    TopK top(2);
    CHECK(top.size() == 0 && !top.full() && std::isinf(top.worst_distance()));
    top.push({3.0f, 1});
    CHECK(top.size() == 1 && !top.full() && top.worst_distance() == 3.0f);
    top.push({1.0f, 2});
    CHECK(top.full() && top.worst_distance() == 3.0f);
    top.push({2.0f, 3});
    CHECK(top.worst_distance() == 2.0f);
}

TEST(helpers, topk_rejects_worse) {
    TopK top(2);
    top.push({1.0f, 1});
    top.push({2.0f, 2});
    CHECK(!top.push({7.0f, 3}));
    CHECK(top.push({0.5f, 4}));
}

TEST(helpers, topk_ties_by_id) {
    TopK ties(2);
    ties.push({1.0f, 5});
    ties.push({1.0f, 7});
    CHECK(ties.push({1.0f, 6}));   // beats {1, 7}
    CHECK(!ties.push({1.0f, 9}));  // loses to both
    auto t = ties.take_sorted();
    CHECK(t[0].id == 5 && t[1].id == 6);
}

TEST(helpers, topk_k_zero) {
    TopK none(0);
    CHECK(!none.push({1.0f, 1}));
    CHECK(none.take_sorted().empty() && none.full());
}

TEST(helpers, topk_k_one) {
    TopK one(1);
    for (NodeId i = 0; i < 10; ++i) one.push({float(10 - i), i});
    auto best = one.take_sorted();
    CHECK(best.size() == 1 && best[0].id == 9);
}

TEST(helpers, topk_huge_k) {
    TopK huge(kHuge);
    for (NodeId i = 0; i < 3; ++i) huge.push({float(i), i});
    CHECK(huge.size() == 3 && !huge.full());
    CHECK(huge.take_sorted().size() == 3);
}

TEST(helpers, topk_infinity) {
    TopK top(1);
    top.push({kInf, 1});
    top.push({1.0f, 2});
    CHECK(top.take_sorted()[0].id == 2);
    TopK all_inf(1);
    all_inf.push({kInf, 8});
    all_inf.push({kInf, 3});
    CHECK(all_inf.take_sorted()[0].id == 3);  // tie: smaller id
}

TEST(helpers, topk_take_empties_and_reuse) {
    TopK top(2);
    top.push({1.0f, 1});
    CHECK(top.take_sorted().size() == 1);
    CHECK(top.take_sorted().empty() && top.size() == 0);
    top.push({5.0f, 9});
    CHECK(top.size() == 1 && top.take_sorted()[0].id == 9);
}

TEST(helpers, topk_matches_full_sort) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<Candidate> all;
    TopK top(37);
    for (NodeId i = 0; i < 10000; ++i) {
        Candidate c{dist(rng), i};
        all.push_back(c);
        top.push(c);
    }
    std::sort(all.begin(), all.end());
    auto got = top.take_sorted();
    REQUIRE(got.size() == 37);
    bool same = true;
    for (std::size_t i = 0; i < 37; ++i) same = same && got[i].id == all[i].id;
    CHECK(same);
}

// ===========================================================================
// Layer 3 helpers: PreparedVector
// ===========================================================================

TEST(helpers, prepared_alignment_and_stride) {
    PreparedVector p(5), one(1), big(100);
    CHECK(p.stride() == 16 && p.dim() == 5 && aligned(p.data()));
    CHECK(one.stride() == 16 && big.stride() == 112);
    CHECK(p.values().size() == 5 && p.padded().size() == 16);
}

TEST(helpers, prepared_l2_and_ip_copy_unchanged) {
    PreparedVector p(3);
    p.prepare(std::vector<float>{3, 4, 5}, Metric::L2);
    CHECK(p.values()[0] == 3.0f && p.values()[2] == 5.0f);
    p.prepare(std::vector<float>{3, 4, 5}, Metric::InnerProduct);
    CHECK(p.values()[0] == 3.0f && p.values()[2] == 5.0f);
}

TEST(helpers, prepared_cosine_normalized) {
    PreparedVector p(2);
    p.prepare(std::vector<float>{3, 4}, Metric::Cosine);
    CHECK(std::fabs(p.values()[0] - 0.6f) < 1e-6f && std::fabs(p.values()[1] - 0.8f) < 1e-6f);
}

TEST(helpers, prepared_padding_zero_after_reuse) {
    PreparedVector p(5);
    for (int i = 0; i < 10; ++i) p.prepare(std::vector<float>(5, float(i + 1)), Metric::L2);
    bool zero = true;
    for (std::size_t i = 5; i < p.padded().size(); ++i) zero = zero && p.padded()[i] == 0.0f;
    CHECK(zero);
}

TEST(helpers, prepared_rejects_wrong_dim_unchanged) {
    PreparedVector p(3);
    p.prepare(std::vector<float>{1, 2, 3}, Metric::L2);
    CHECK_THROWS_AS(std::invalid_argument, p.prepare(std::vector<float>{9, 9}, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, p.prepare(std::vector<float>{9, 9, 9, 9}, Metric::L2));
    CHECK(p.values()[0] == 1.0f && p.values()[2] == 3.0f);
}

TEST(helpers, prepared_rejects_nan_inf_unchanged) {
    PreparedVector p(3);
    p.prepare(std::vector<float>{1, 2, 3}, Metric::L2);
    for (float bad : {kNaN, kInf, -kInf})
        for (Metric m : kMetrics)
            CHECK_THROWS_AS(std::invalid_argument, p.prepare(std::vector<float>{1, bad, 3}, m));
    CHECK(p.values()[0] == 1.0f && p.values()[1] == 2.0f && p.values()[2] == 3.0f);
}

TEST(helpers, prepared_rejects_zero_dim) {
    CHECK_THROWS_AS(std::invalid_argument, PreparedVector bad(0));
}

TEST(helpers, prepared_zero_vector_cosine) {
    PreparedVector p(3);
    p.prepare(std::vector<float>{0, 0, 0}, Metric::Cosine);
    bool zero = true;
    for (float x : p.values()) zero = zero && x == 0.0f;
    CHECK(zero);
}

TEST(helpers, prepared_tiny_vector_cosine) {
    PreparedVector p(3);
    p.prepare(std::vector<float>{1e-40f, 0, 0}, Metric::Cosine);
    CHECK(std::fabs(p.values()[0] - 1.0f) < 1e-6f);
}

TEST(helpers, prepared_finite_extremes_accepted) {
    PreparedVector p(3);
    const float max = std::numeric_limits<float>::max();
    CHECK_NOTHROW(p.prepare(std::vector<float>{max, -max, -0.0f}, Metric::L2));
    CHECK(p.values()[0] == max && std::signbit(p.values()[2]));
    CHECK_NOTHROW(p.prepare(std::vector<float>{std::numeric_limits<float>::denorm_min(), 0, 0},
                            Metric::L2));
}

// ===========================================================================
// Layer 3 helpers: VisitedList and VisitedListPool
// ===========================================================================

TEST(helpers, visited_basic) {
    VisitedList v;
    v.reset(10);
    CHECK(v.visit(3));
    CHECK(!v.visit(3));
    CHECK(v.visited(3) && !v.visited(4));
}

TEST(helpers, visited_reset_clears) {
    VisitedList v;
    v.reset(10);
    v.visit(3);
    v.reset(10);
    CHECK(!v.visited(3) && v.visit(3));
}

TEST(helpers, visited_growth) {
    VisitedList v;
    v.reset(0);
    v.reset(5);
    CHECK(v.visit(4));
    v.reset(20);
    CHECK(v.visit(19) && !v.visited(4));
}

TEST(helpers, visited_epoch_wrap) {
    VisitedList v;
    v.reset(20);
    v.set_epoch_for_testing(0xFFFFFFFFu);
    v.visit(7);
    v.reset(20);  // wraps: array cleared, epoch restarts at 1
    CHECK(!v.visited(7) && !v.visited(19));
    CHECK(v.visit(7));
}

TEST(helpers, visited_many_resets) {
    VisitedList v;
    bool ok = true;
    for (int i = 0; i < 1000; ++i) {
        v.reset(5);
        ok = ok && !v.visited(4);
        v.visit(4);
    }
    CHECK(ok);
}

TEST(helpers, pool_independent_lists) {
    VisitedListPool pool;
    auto a = pool.acquire(10);
    auto b = pool.acquire(10);
    CHECK(&*a != &*b);
    CHECK(a->visit(1) && b->visit(1));
    CHECK(pool.idle_count() == 0);
}

TEST(helpers, pool_reuse_and_reset) {
    VisitedListPool pool;
    {
        auto a = pool.acquire(10);
        a->visit(1);
    }
    CHECK(pool.idle_count() == 1);
    auto b = pool.acquire(10);
    CHECK(pool.idle_count() == 0 && !b->visited(1));
}

TEST(helpers, pool_returns_on_exception) {
    VisitedListPool pool;
    try {
        auto h = pool.acquire(10);
        throw std::runtime_error("search failed");
    } catch (const std::runtime_error&) {
    }
    CHECK(pool.idle_count() == 1);
}

TEST(helpers, pool_reused_list_grows) {
    VisitedListPool pool;
    { auto h = pool.acquire(10); }
    auto h = pool.acquire(1000);
    CHECK(h->visit(999));
}

TEST(helpers, pool_concurrent_acquire) {
    VisitedListPool pool;
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 2000; ++i) {
                auto h = pool.acquire(64);
                if (!h->visit(NodeId(i % 64))) ++errors;  // fresh search: first visit
                if (h->visit(NodeId(i % 64))) ++errors;   // second visit
            }
        });
    for (auto& t : threads) t.join();
    CHECK(errors.load() == 0);
    CHECK(pool.idle_count() >= 1 && pool.idle_count() <= 8);
}

// ===========================================================================
// FlatIndex
// ===========================================================================

TEST(flat, rejects_zero_dim) {
    CHECK_THROWS_AS(std::invalid_argument, FlatIndex bad(0, Metric::L2));
}

TEST(flat, accessors) {
    FlatIndex f(7, Metric::Cosine);
    CHECK(f.dim() == 7 && f.metric() == Metric::Cosine && f.size() == 0);
}

TEST(flat, matches_reference_all_metrics) {
    const std::size_t dim = 37;  // not a multiple of 16, so padding is exercised
    auto data = random_vectors(500, dim, 1);
    auto queries = random_vectors(20, dim, 2);
    std::vector<bool> removed(data.size(), false);
    for (Metric m : kMetrics) {
        FlatIndex f(dim, m);
        for (std::size_t i = 0; i < data.size(); ++i) f.add(1000 + i, data[i]);
        for (const auto& q : queries)
            for (std::size_t k : std::initializer_list<std::size_t>{1, 10, 50, 500})
                CHECK(same_results(f.search(q, k), reference_search(data, removed, q, m, k, 1000)));
    }
}

TEST(flat, results_sorted_unique_live) {
    auto data = random_vectors(300, 16, 3);
    for (Metric m : kMetrics) {
        FlatIndex f(16, m);
        for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
        for (std::size_t i = 0; i < 10; ++i) check_results(f, f.search(data[i], 50));
    }
}

TEST(flat, self_query_is_first) {
    auto data = random_vectors(200, 24, 4);
    FlatIndex f(24, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    bool ok = true;
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto r = f.search(data[i], 1);
        ok = ok && r[0].id == i && r[0].distance == 0.0f;
    }
    CHECK(ok);
}

TEST(flat, dim_one_ties_by_insertion) {
    FlatIndex f(1, Metric::L2);
    f.add(5, std::vector<float>{5});
    f.add(1, std::vector<float>{1});
    f.add(3, std::vector<float>{3});
    auto r = f.search(std::vector<float>{2}, 3);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 1 && r[1].id == 3 && r[2].id == 5);  // 1 and 3 tie at distance 1
    CHECK(r[0].distance == 1.0f && r[1].distance == 1.0f && r[2].distance == 9.0f);
}

TEST(flat, identical_vectors_in_insertion_order) {
    FlatIndex f(4, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) f.add(100 - i, std::vector<float>{1, 1, 1, 1});
    auto r = f.search(std::vector<float>{1, 1, 1, 1}, 10);
    bool ok = r.size() == 10;
    for (std::size_t i = 0; ok && i < 10; ++i) ok = r[i].id == 100 - i && r[i].distance == 0.0f;
    CHECK(ok);
}

TEST(flat, empty_index) {
    FlatIndex f(4, Metric::L2);
    CHECK(f.search(std::vector<float>{1, 2, 3, 4}, 5).empty());
    CHECK(!f.contains(0) && !f.remove(0));
}

TEST(flat, k_zero) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    CHECK(f.search(std::vector<float>{0, 0}, 0).empty());
}

TEST(flat, k_larger_than_size) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    f.add(2, std::vector<float>{1, 1});
    auto r = f.search(std::vector<float>{0, 0}, 100);
    CHECK(r.size() == 2 && r[0].id == 1 && r[1].id == 2);
}

TEST(flat, huge_k) {
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) f.add(i, std::vector<float>{float(i), 0});
    CHECK(f.search(std::vector<float>{0, 0}, kHuge).size() == 10);
}

TEST(flat, wrong_dim_rejected) {
    FlatIndex f(3, Metric::L2);
    CHECK_THROWS_AS(std::invalid_argument, f.search(std::vector<float>(2), 1));  // even when empty
    CHECK_THROWS_AS(std::invalid_argument, f.add(1, std::vector<float>(2)));
    CHECK_THROWS_AS(std::invalid_argument, f.add(1, std::vector<float>(4)));
    f.add(1, std::vector<float>(3, 0.0f));
    CHECK_THROWS_AS(std::invalid_argument, f.search(std::vector<float>(4), 1));
    CHECK(f.size() == 1);
}

TEST(flat, nan_inf_rejected) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    for (float bad : {kNaN, kInf, -kInf}) {
        CHECK_THROWS_AS(std::invalid_argument, f.add(2, std::vector<float>{bad, 0}));
        CHECK_THROWS_AS(std::invalid_argument, f.search(std::vector<float>{0, bad}, 1));
    }
    CHECK(f.size() == 1 && !f.contains(2));
    CHECK(f.search(std::vector<float>{0, 0}, 1)[0].id == 1);
}

TEST(flat, duplicate_rejected) {
    FlatIndex f(2, Metric::L2);
    f.add(7, std::vector<float>{0, 0});
    CHECK_THROWS_AS(std::invalid_argument, f.add(7, std::vector<float>{5, 5}));
    CHECK(f.size() == 1);
    CHECK(f.search(std::vector<float>{0, 0}, 1)[0].distance == 0.0f);  // original kept
}

TEST(flat, removed_id_reusable) {
    // Changed by real deletion: a removed ID can be added again.
    FlatIndex f(2, Metric::L2);
    f.add(7, std::vector<float>{0, 0});
    f.remove(7);
    CHECK_NOTHROW(f.add(7, std::vector<float>{5, 5}));
    CHECK(f.size() == 1 && f.contains(7));
    auto r = f.search(std::vector<float>{5, 5}, 1);
    CHECK(r[0].id == 7 && r[0].distance == 0.0f);
}

TEST(flat, remove_semantics) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    f.add(2, std::vector<float>{1, 1});
    CHECK(f.contains(1) && f.size() == 2);
    CHECK(f.remove(1));
    CHECK(!f.contains(1) && f.contains(2) && f.size() == 1);
    CHECK(!f.remove(1));     // already removed
    CHECK(!f.remove(999));   // never existed
    CHECK(f.size() == 1);
}

TEST(flat, removed_never_returned) {
    auto data = random_vectors(200, 16, 3);
    auto queries = random_vectors(10, 16, 4);
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(1000 + i, data[i]);
    std::vector<bool> removed(data.size(), false);
    for (std::size_t i = 0; i < data.size(); i += 3) {
        f.remove(1000 + i);
        removed[i] = true;
    }
    CHECK(f.size() == 200 - 67);
    for (const auto& q : queries) {
        auto got = f.search(q, 20);
        // Swap-with-last reorders internal positions, so compare tie-tolerantly.
        CHECK(same_results(tie_sorted(got), tie_sorted(reference_search(data, removed, q, Metric::L2, 20, 1000))));
        check_results(f, got);
    }
}

TEST(flat, remove_all_then_add) {
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 5; ++i) f.add(i, std::vector<float>{float(i), 0});
    for (std::uint64_t i = 0; i < 5; ++i) CHECK(f.remove(i));
    CHECK(f.size() == 0 && f.search(std::vector<float>{0, 0}, 3).empty());
    f.add(100, std::vector<float>{9, 9});
    auto r = f.search(std::vector<float>{0, 0}, 3);
    CHECK(r.size() == 1 && r[0].id == 100);
}

TEST(flat, inner_product_order) {
    FlatIndex f(2, Metric::InnerProduct);
    for (std::uint64_t i = 1; i <= 3; ++i) f.add(i, std::vector<float>{float(i), 0});
    auto r = f.search(std::vector<float>{1, 0}, 3);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 3 && r[0].distance == -3.0f);
    CHECK(r[1].id == 2 && r[1].distance == -2.0f);
    CHECK(r[2].id == 1 && r[2].distance == -1.0f);
}

TEST(flat, cosine_ignores_length) {
    FlatIndex f(3, Metric::Cosine);
    f.add(1, std::vector<float>{1, 0, 0});
    f.add(2, std::vector<float>{0, 5, 0});
    auto r = f.search(std::vector<float>{0, 0.1f, 0}, 2);
    CHECK(r[0].id == 2 && std::fabs(r[0].distance) < 1e-6f);
    CHECK(r[1].id == 1 && std::fabs(r[1].distance - 1.0f) < 1e-6f);
}

TEST(flat, cosine_zero_vectors) {
    FlatIndex f(2, Metric::Cosine);
    f.add(1, std::vector<float>{3, 4});
    f.add(2, std::vector<float>{0, 0});
    auto r = f.search(std::vector<float>{0, 0}, 2);
    CHECK(r[0].distance == 1.0f && r[1].distance == 1.0f);  // zero query: 1 to everything
    r = f.search(std::vector<float>{6, 8}, 2);
    CHECK(r[0].id == 1 && r[1].id == 2 && r[1].distance == 1.0f);  // zero stored: 1
}

TEST(flat, extreme_ids) {
    FlatIndex f(2, Metric::L2);
    f.add(0, std::vector<float>{0, 0});
    f.add(kMaxId, std::vector<float>{1, 1});
    CHECK(f.search(std::vector<float>{1, 1}, 1)[0].id == kMaxId);
    CHECK(f.search(std::vector<float>{0, 0}, 1)[0].id == 0);
    CHECK(f.remove(kMaxId) && !f.contains(kMaxId));
}

TEST(flat, high_dimension) {
    auto data = random_vectors(30, 1536, 5);
    std::vector<bool> removed(30, false);
    for (Metric m : kMetrics) {
        FlatIndex f(1536, m);
        for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
        CHECK(same_results(f.search(data[3], 10), reference_search(data, removed, data[3], m, 10, 0)));
    }
}

TEST(flat, many_vectors) {
    auto data = random_vectors(5000, 8, 6);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    CHECK(f.size() == 5000);
    auto r = f.search(data[4321], 5);
    CHECK(r[0].id == 4321 && r.size() == 5);
    check_results(f, r);
}

// ===========================================================================
// HnswIndex: construction and input validation
// ===========================================================================

TEST(hnsw, rejects_invalid_params) {
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(0, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{0, 200, 1}));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{1, 200, 1}));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{16, 0, 1}));
    CHECK_NOTHROW(HnswIndex ok(4, Metric::L2, HnswParams{2, 1, 1}));
}

TEST(hnsw, accessors) {
    HnswIndex h(7, Metric::InnerProduct, HnswParams{12, 50, 9});
    CHECK(h.dim() == 7 && h.metric() == Metric::InnerProduct && h.size() == 0);
    CHECK(h.params().M == 12 && h.params().ef_construction == 50 && h.params().seed == 9);
}

TEST(hnsw, empty_index) {
    HnswIndex h(4, Metric::L2);
    CHECK(h.search(std::vector<float>{1, 2, 3, 4}, 5).empty());
    CHECK(h.max_level() == -1 && h.entry_point() == kEmpty);
    CHECK(!h.contains(1) && !h.remove(1));
    check_graph(h, 1.0);
}

TEST(hnsw, single_vector) {
    HnswIndex h(4, Metric::L2);
    h.add(7, std::vector<float>{1, 2, 3, 4});
    auto r = h.search(std::vector<float>{0, 0, 0, 0}, 3);
    CHECK(r.size() == 1 && r[0].id == 7);
    CHECK(h.entry_point() == 0 && h.size() == 1);
    check_graph(h, 1.0);
}

TEST(hnsw, wrong_dim_rejected) {
    HnswIndex h(3, Metric::L2);
    CHECK_THROWS_AS(std::invalid_argument, h.search(std::vector<float>(2), 1));  // even when empty
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, std::vector<float>(2)));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, std::vector<float>(4)));
    h.add(1, std::vector<float>(3, 0.0f));
    CHECK_THROWS_AS(std::invalid_argument, h.search(std::vector<float>(4), 1));
    CHECK(h.size() == 1 && h.storage().size() == 1);
}

TEST(hnsw, nan_inf_rejected_unchanged) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    h.add(2, std::vector<float>{1, 1});
    const NodeId entry = h.entry_point();
    const int top = h.max_level();
    for (float bad : {kNaN, kInf, -kInf}) {
        CHECK_THROWS_AS(std::invalid_argument, h.add(3, std::vector<float>{bad, 0}));
        CHECK_THROWS_AS(std::invalid_argument, h.search(std::vector<float>{0, bad}, 1));
    }
    CHECK(h.size() == 2 && h.storage().size() == 2 && !h.contains(3));
    CHECK(h.entry_point() == entry && h.max_level() == top);
}

TEST(hnsw, duplicate_rejected) {
    HnswIndex h(2, Metric::L2);
    h.add(7, std::vector<float>{0, 0});
    CHECK_THROWS_AS(std::invalid_argument, h.add(7, std::vector<float>{5, 5}));
    CHECK(h.size() == 1 && h.storage().size() == 1);
    CHECK(h.search(std::vector<float>{0, 0}, 1)[0].distance == 0.0f);
}

TEST(hnsw, removed_id_reusable) {
    // Changed by real deletion: a removed ID can be added again.
    HnswIndex h(2, Metric::L2);
    h.add(7, std::vector<float>{0, 0});
    h.add(8, std::vector<float>{9, 9});
    h.remove(7);
    CHECK_NOTHROW(h.add(7, std::vector<float>{5, 5}));
    CHECK(h.size() == 2 && h.contains(7));
    auto r = h.search(std::vector<float>{5, 5}, 1);
    CHECK(r[0].id == 7 && r[0].distance == 0.0f);
}

TEST(hnsw, failed_inserts_keep_graph_identical) {
    // Rejected inserts must not consume random numbers, or later levels would differ.
    auto data = clustered(300, 8, 5, 21);
    HnswIndex clean(8, Metric::L2), noisy(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        clean.add(i, data[i]);
        noisy.add(i, data[i]);
        if (i > 0 && i % 10 == 0) {
            try { noisy.add(0, data[i]); } catch (...) {}                         // duplicate
            try { noisy.add(9999, std::vector<float>(8, kNaN)); } catch (...) {}  // NaN
            try { noisy.add(9998, std::vector<float>(3)); } catch (...) {}        // bad dim
        }
    }
    CHECK(clean.entry_point() == noisy.entry_point() && clean.max_level() == noisy.max_level());
    bool same = true;
    for (std::size_t i = 0; i < 30; ++i) same = same && same_results(clean.search(data[i], 10), noisy.search(data[i], 10));
    CHECK(same);
}

// ===========================================================================
// HnswIndex: search behavior
// ===========================================================================

TEST(hnsw, tiny_indexes_match_flat) {
    for (std::size_t n = 1; n <= 10; ++n) {
        HnswIndex h(4, Metric::L2);
        FlatIndex f(4, Metric::L2);
        fill(h, f, random_vectors(n, 4, unsigned(30 + n)));
        auto q = random_vectors(1, 4, 99)[0];
        CHECK(same_results(h.search(q, 20, 20), f.search(q, 20)));
        check_graph(h, 1.0);
    }
}

TEST(hnsw, dim_one) {
    HnswIndex h(1, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i, std::vector<float>{float(i)});
    auto r = h.search(std::vector<float>{50.2f}, 3, 50);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 50 && r[1].id == 51 && r[2].id == 49);
}

TEST(hnsw, k_zero) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    CHECK(h.search(std::vector<float>{0, 0}, 0).empty());
}

TEST(hnsw, k_larger_than_size) {
    HnswIndex h(4, Metric::L2);
    auto data = random_vectors(6, 4, 41);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    CHECK(h.search(data[0], 100, 1).size() == 6);  // ef < k is raised to k
}

TEST(hnsw, huge_k_and_ef) {
    HnswIndex h(4, Metric::L2);
    auto data = random_vectors(50, 4, 40);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    auto r = h.search(data[0], kHuge, kHuge);
    CHECK(r.size() == 50);
    check_results(h, r);
}

TEST(hnsw, ef_zero_raised_to_k) {
    HnswIndex h(4, Metric::L2);
    auto data = random_vectors(50, 4, 40);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    CHECK(h.search(data[0], 5, 0).size() == 5);
}

TEST(hnsw, results_sorted_unique_live) {
    for (Metric m : kMetrics) {
        const Fixture& fx = fixture(m);
        for (std::size_t i = 0; i < 30; ++i) check_results(*fx.hnsw, fx.hnsw->search(fx.queries[i], 10, 64));
    }
}

TEST(hnsw, self_query_found) {
    for (Metric m : {Metric::L2, Metric::Cosine}) {  // not inner product: self need not be closest
        const Fixture& fx = fixture(m);
        bool ok = true;
        for (std::size_t i = 0; i < 50; ++i) {
            auto r = fx.hnsw->search(fx.data[i], 1, 64);
            ok = ok && r.size() == 1 && r[0].id == i && std::fabs(r[0].distance) < 1e-5f;
        }
        CHECK(ok);
    }
}

TEST(hnsw, distances_equal_flat) {
    for (Metric m : kMetrics) {
        const Fixture& fx = fixture(m);
        bool ok = true;
        for (std::size_t i = 0; i < 10; ++i) {
            Results all = fx.flat->search(fx.queries[i], kHuge);
            for (const auto& x : fx.hnsw->search(fx.queries[i], 10, 64)) {
                auto it = std::find_if(all.begin(), all.end(), [&](const SearchResult& y) { return y.id == x.id; });
                ok = ok && it != all.end() && it->distance == x.distance;
            }
        }
        CHECK(ok);
    }
}

TEST(hnsw, inner_product_order) {
    HnswIndex h(2, Metric::InnerProduct);
    for (std::uint64_t i = 1; i <= 20; ++i) h.add(i, std::vector<float>{float(i), 0});
    auto r = h.search(std::vector<float>{1, 0}, 3, 50);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 20 && r[1].id == 19 && r[2].id == 18);
    CHECK(r[0].distance == -20.0f);
}

TEST(hnsw, cosine_ignores_length) {
    HnswIndex h(4, Metric::Cosine);
    auto data = random_vectors(100, 4, 61);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    std::vector<float> scaled = data[7];
    for (float& x : scaled) x *= 1000.0f;
    auto r = h.search(scaled, 1, 50);
    CHECK(r[0].id == 7 && std::fabs(r[0].distance) < 1e-5f);
}

TEST(hnsw, deterministic_same_seed) {
    auto data = clustered(800, 16, 8, 6);
    HnswIndex a(16, Metric::L2), b(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        a.add(i, data[i]);
        b.add(i, data[i]);
    }
    CHECK(a.entry_point() == b.entry_point() && a.max_level() == b.max_level());
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) same = same && same_results(a.search(data[i * 7], 10), b.search(data[i * 7], 10));
    CHECK(same);
}

TEST(hnsw, repeated_search_identical) {
    const Fixture& fx = fixture(Metric::L2);
    auto first = fx.hnsw->search(fx.queries[0], 10, 64);
    bool same = true;
    for (int i = 0; i < 20; ++i) same = same && same_results(first, fx.hnsw->search(fx.queries[0], 10, 64));
    CHECK(same);
}

// ===========================================================================
// HnswIndex: graph structure and accuracy
// ===========================================================================

TEST(hnsw, graph_valid_all_metrics) {
    for (Metric m : kMetrics) check_graph(*fixture(m).hnsw, 0.99);
}

TEST(hnsw, recall_l2) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK(recall(*fx.hnsw, *fx.flat, fx.queries, 10, 100) >= 0.95);
}

TEST(hnsw, recall_cosine) {
    const Fixture& fx = fixture(Metric::Cosine);
    CHECK(recall(*fx.hnsw, *fx.flat, fx.queries, 10, 100) >= 0.95);
}

TEST(hnsw, recall_inner_product) {
    const Fixture& fx = fixture(Metric::InnerProduct);
    CHECK(recall(*fx.hnsw, *fx.flat, fx.queries, 10, 100) >= 0.95);
}

TEST(hnsw, recall_grows_with_ef) {
    const Fixture& fx = fixture(Metric::L2);
    const double low = recall(*fx.hnsw, *fx.flat, fx.queries, 10, 10);
    const double high = recall(*fx.hnsw, *fx.flat, fx.queries, 10, 400);
    CHECK(high >= low);
    CHECK(high >= 0.99);
}

TEST(hnsw, recall_at_k1_and_k50) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK(recall(*fx.hnsw, *fx.flat, fx.queries, 1, 64) >= 0.95);
    CHECK(recall(*fx.hnsw, *fx.flat, fx.queries, 50, 200) >= 0.95);
}

TEST(hnsw, level_distribution) {
    HnswIndex h(16, Metric::L2, HnswParams{16, 32, 7});
    auto data = clustered(4000, 16, 4, 8);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    std::size_t upper = 0;
    for (NodeId n = 0; n < 4000; ++n) upper += h.storage().graph().level(n) > 0;
    const double fraction = double(upper) / 4000.0;  // expected 1/16 = 0.0625
    CHECK(fraction > 0.045 && fraction < 0.08);
    CHECK(h.max_level() >= 1);
}

TEST(hnsw, storage_in_sync) {
    const Fixture& fx = fixture(Metric::L2);
    const Storage& st = fx.hnsw->storage();
    CHECK(st.size() == 3000 && st.ids().size() == 3000 && st.graph().size() == 3000);
    CHECK(fx.hnsw->size() == 3000);
}

TEST(hnsw, param_minimal_M) {
    // M = 2 builds a valid but sparse graph (measured: ~90% reachable, recall ~0.74).
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    HnswIndex h(16, Metric::L2, HnswParams{2, 200, 1});
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    check_graph(h, 0.80);
    CHECK(recall(h, f, queries, 10, 100) >= 0.60);
}

TEST(hnsw, param_minimal_ef_construction) {
    // ef_construction = 1 builds greedily (measured recall ~0.66).
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    HnswIndex h(16, Metric::L2, HnswParams{16, 1, 1});
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    check_graph(h, 0.99);
    CHECK(recall(h, f, queries, 10, 100) >= 0.50);
}

TEST(hnsw, param_large_M) {
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    HnswIndex h(16, Metric::L2, HnswParams{64, 200, 1});
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    check_graph(h, 0.999);
    CHECK(recall(h, f, queries, 10, 100) >= 0.99);
}

TEST(hnsw, param_other_seeds) {
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    for (std::uint64_t seed : {0ull, 777ull, 123456789ull}) {
        HnswIndex h(16, Metric::L2, HnswParams{16, 200, seed});
        for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
        check_graph(h, 0.99);
        CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    }
}

TEST(hnsw, identical_vectors_harmless) {
    // Not every copy can stay reachable (fixed link slots), but copies must not
    // damage the rest of the graph.
    HnswIndex h(8, Metric::L2);
    std::vector<float> same(8, 0.5f);
    auto noise = random_vectors(200, 8, 60);
    for (std::uint64_t i = 0; i < 200; ++i) {
        h.add(i, same);             // IDs 0..199: identical
        h.add(1000 + i, noise[i]);  // IDs 1000..1199: random
    }
    auto r = h.search(same, 10, 100);
    CHECK(r.size() == 10);
    bool copies = true;
    for (const auto& x : r) copies = copies && x.id < 200 && x.distance == 0.0f;
    CHECK(copies);
    check_graph(h, 0.70);
    bool found = true;
    for (std::uint64_t i = 0; i < 200; ++i) {
        auto self = h.search(noise[i], 1, 100);
        found = found && self.size() == 1 && self[0].id == 1000 + i && self[0].distance == 0.0f;
    }
    CHECK(found);
}

TEST(hnsw, all_identical_vectors) {
    HnswIndex h(4, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i, std::vector<float>{1, 2, 3, 4});
    auto r = h.search(std::vector<float>{1, 2, 3, 4}, 5, 50);
    CHECK(!r.empty());
    for (const auto& x : r) CHECK(x.distance == 0.0f);
    check_results(h, r);
}

// ===========================================================================
// HnswIndex: removal
// ===========================================================================

TEST(hnsw, remove_semantics) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    h.add(2, std::vector<float>{1, 1});
    CHECK(h.remove(1));
    CHECK(!h.contains(1) && h.contains(2) && h.size() == 1);
    CHECK(!h.remove(1) && !h.remove(999));
    CHECK(h.capacity() == 2 && h.deleted_count() == 1);  // slot kept for reuse
    h.add(3, std::vector<float>{2, 2});
    CHECK(h.capacity() == 2 && h.deleted_count() == 0);  // ...and reused
}

TEST(hnsw, removed_never_returned) {
    auto data = clustered(400, 8, 4, 70);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::size_t i = 0; i < 400; i += 5) h.remove(i);
    bool ok = true;
    for (std::size_t i = 0; i < 400; i += 5)
        for (const auto& x : h.search(data[i], 10, 100)) ok = ok && x.id % 5 != 0;
    CHECK(ok);
}

TEST(hnsw, remove_entry_point) {
    auto data = clustered(500, 8, 4, 71);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    const std::uint64_t entry_id = h.storage().ids().external(h.entry_point());
    CHECK(h.remove(entry_id));
    f.remove(entry_id);
    CHECK(!h.contains(entry_id));
    CHECK(recall(h, f, clustered(30, 8, 4, 72), 10, 100) >= 0.95);
}

TEST(hnsw, remove_quarter_recall) {
    auto data = clustered(2000, 32, 10, 4);
    auto queries = clustered(50, 32, 10, 5);
    HnswIndex h(32, Metric::L2);
    FlatIndex f(32, Metric::L2);
    fill(h, f, data);
    for (std::size_t i = 0; i < data.size(); i += 4) {
        h.remove(i);
        f.remove(i);
    }
    CHECK(h.size() == f.size() && h.size() == 1500);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    for (const auto& q : queries) check_results(h, h.search(q, 10, 100));
}

TEST(hnsw, remove_all_but_one) {
    auto data = clustered(400, 8, 4, 70);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 400; ++i)
        if (i != 123) h.remove(i);
    CHECK(h.size() == 1);
    auto r = h.search(data[0], 10, 400);
    CHECK(r.size() == 1 && r[0].id == 123);
}

TEST(hnsw, remove_all_then_add) {
    auto data = clustered(200, 8, 4, 73);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 200; ++i) h.remove(i);
    CHECK(h.size() == 0 && h.search(data[0], 5).empty());
    CHECK(h.entry_point() == kEmpty && h.max_level() == -1);  // as in a new index
    CHECK(h.capacity() == 200);                               // slots kept for reuse
    h.add(5000, data[0]);
    h.add(5001, data[1]);
    CHECK(h.capacity() == 200);  // reused, not appended
    auto r = h.search(data[0], 5, 100);
    CHECK(r.size() == 2 && r[0].id == 5000);
    check_graph(h, 1.0);
}

TEST(hnsw, new_nodes_do_not_link_to_removed) {
    // Rewritten for slot reuse: new vectors take freed slots, so they are found
    // through their user IDs. 100 removed, 50 slots reused, 50 stay removed.
    auto data = clustered(350, 16, 6, 74);
    HnswIndex h(16, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) h.add(i, data[i]);
    for (std::size_t i = 0; i < 300; i += 3) h.remove(i);
    for (std::size_t i = 300; i < 350; ++i) h.add(i, data[i]);
    CHECK(h.capacity() == 300 && h.deleted_count() == 50);
    const Storage& st = h.storage();
    bool ok = true;
    for (std::uint64_t id = 300; id < 350; ++id) {
        const NodeId node = *st.ids().find(id);
        for (int l = 0; l <= st.graph().level(node); ++l)
            for (NodeId nb : st.graph().links(node, l)) {
                if (nb == kEmpty) break;
                ok = ok && !st.ids().is_deleted(nb);
            }
    }
    CHECK(ok);
    check_graph(h, 0.99);
}

TEST(hnsw, remove_and_readd_recall) {
    auto data = clustered(1500, 16, 8, 75);
    auto queries = clustered(40, 16, 8, 76);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < 1000; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (std::size_t i = 0; i < 1000; i += 2) {
        h.remove(i);
        f.remove(i);
    }
    for (std::size_t i = 1000; i < 1500; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(h.size() == 1000);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    check_graph(h, 0.99);
}

// ===========================================================================
// Robustness: numeric extremes
// ===========================================================================

TEST(robustness, ip_overflow_flat_has_no_nan) {
    // Finite but huge values: an inner product can overflow to +inf and -inf,
    // whose sum is NaN. Such a pair must sort last instead of breaking the order.
    FlatIndex f(2, Metric::InnerProduct);
    f.add(1, std::vector<float>{1, 1});
    f.add(2, std::vector<float>{3e38f, -3e38f});  // with the query: +inf + -inf = NaN
    f.add(3, std::vector<float>{2, 2});
    auto r = f.search(std::vector<float>{3e38f, 3e38f}, 3);
    REQUIRE(r.size() == 3);
    for (const auto& x : r) CHECK(!std::isnan(x.distance));
    CHECK(r[2].id == 2 && std::isinf(r[2].distance) && r[2].distance > 0);
    check_results(f, r);
}

TEST(robustness, ip_overflow_hnsw_has_no_nan) {
    HnswIndex h(2, Metric::InnerProduct);
    auto data = random_vectors(200, 2, 81);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 20; ++i)
        h.add(1000 + i, std::vector<float>{3e38f, (i % 2 ? 1.0f : -1.0f) * 3e38f});
    for (float s : {1.0f, -1.0f}) {
        auto r = h.search(std::vector<float>{3e38f, s * 3e38f}, 30, 100);
        bool no_nan = true;
        for (const auto& x : r) no_nan = no_nan && !std::isnan(x.distance);
        CHECK(no_nan);
        check_results(h, r);
    }
    check_graph(h, 0.90);
}

TEST(robustness, l2_overflow_sorted_as_infinity) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{-3e38f, 0});
    f.add(2, std::vector<float>{3e38f, 1});  // close to the query: distance 1
    f.add(3, std::vector<float>{-3e38f, 1});
    auto r = f.search(std::vector<float>{3e38f, 0}, 3);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 2 && r[0].distance == 1.0f);
    CHECK(r[1].id == 1 && std::isinf(r[1].distance));  // infinities tie: insertion order
    CHECK(r[2].id == 3 && std::isinf(r[2].distance));
}

TEST(robustness, cosine_huge_values) {
    FlatIndex f(3, Metric::Cosine);
    HnswIndex h(3, Metric::Cosine);
    f.add(1, std::vector<float>{1e30f, 1e30f, 0});
    f.add(2, std::vector<float>{0, 1e30f, 1e30f});
    h.add(1, std::vector<float>{1e30f, 1e30f, 0});
    h.add(2, std::vector<float>{0, 1e30f, 1e30f});
    for (const Results& r : {f.search(std::vector<float>{2, 2, 0}, 2), h.search(std::vector<float>{2, 2, 0}, 2)}) {
        REQUIRE(r.size() == 2);
        CHECK(r[0].id == 1 && std::fabs(r[0].distance) < 1e-6f);
        CHECK(std::fabs(r[1].distance - 0.5f) < 1e-6f);  // 60 degrees apart
    }
}

TEST(robustness, tiny_values_all_metrics) {
    const float t = std::numeric_limits<float>::denorm_min();
    for (Metric m : kMetrics) {
        FlatIndex f(2, m);
        HnswIndex h(2, m);
        for (std::uint64_t i = 0; i < 20; ++i) {
            std::vector<float> v{t * float(i), t};
            f.add(i, v);
            h.add(i, v);
        }
        for (const Results& r : {f.search(std::vector<float>{t, t}, 5), h.search(std::vector<float>{t, t}, 5)}) {
            CHECK(r.size() == 5);
            bool finite = true;
            for (const auto& x : r) finite = finite && std::isfinite(x.distance);
            CHECK(finite);
        }
    }
}

TEST(robustness, negative_zero_equals_zero) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0.0f, 0.0f});
    f.add(2, std::vector<float>{-0.0f, -0.0f});
    auto r = f.search(std::vector<float>{0.0f, -0.0f}, 2);
    REQUIRE(r.size() == 2);
    CHECK(r[0].distance == 0.0f && r[1].distance == 0.0f);
}

TEST(robustness, ordered_distance_helper) {
    CHECK(std::isinf(ordered_distance(kNaN)) && ordered_distance(kNaN) > 0);
    CHECK(ordered_distance(1.5f) == 1.5f);
    CHECK(ordered_distance(-kInf) == -kInf && ordered_distance(kInf) == kInf);
}

TEST(robustness, max_level_stays_reasonable) {
    HnswIndex h(8, Metric::L2, HnswParams{16, 16, 11});
    auto data = random_vectors(5000, 8, 82);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    CHECK(h.max_level() >= 1 && h.max_level() <= 8);  // expected about log16(5000) = 3
}

// ===========================================================================
// Robustness: running out of memory
// ===========================================================================

namespace {

/// Runs `op` with allocation n failing, for n = 0, 1, 2, ... After each injected
/// failure, `verify` checks the object is still sound. Stops when `op` completes
/// without any injected failure. Returns the number of allocation points tried,
/// or -1 on a problem (an exception other than bad_alloc, or a failure that the
/// operation swallowed instead of reporting).
template <class Op, class Verify>
long sweep_allocation_failures(Op op, Verify verify, long max_points = 5000) {
    for (long n = 0; n < max_points; ++n) {
        alloc_hook::fail_after(n);
        bool threw = false;
        try {
            op();
        } catch (const std::bad_alloc&) {
            threw = true;
        } catch (const std::exception& e) {
            // Any other exception means an earlier failure corrupted the object.
            alloc_hook::disarm();
            report(false, __LINE__, "operation threw something other than bad_alloc after an earlier failure");
            std::printf("        (at allocation point %ld: %s)\n", n, e.what());
            return -1;
        }
        alloc_hook::disarm();
        if (threw) {
            verify();
        } else if (alloc_hook::fired) {
            // The operation hid an allocation failure and still completed. Code
            // that silently swallows failures is a bug, and the object may have
            // changed, so later points could not be checked against it anyway.
            report(false, __LINE__, "an allocation failure was swallowed instead of reported");
            std::printf("        (at allocation point %ld)\n", n);
            return -1;
        } else {
            return n;  // completed with no failure injected: every point was tried
        }
    }
    return -1;
}

}  // namespace

TEST(robustness, oom_hook_works) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    alloc_hook::fail_after(0);
    CHECK_THROWS_AS(std::bad_alloc, auto p = std::make_unique<int>(1));
    CHECK(alloc_hook::fired);
    CHECK_NOTHROW(auto p = std::make_unique<int>(2));  // fails only once
    alloc_hook::fail_after(2);
    auto a = std::make_unique<int>(1), b = std::make_unique<int>(2);
    CHECK_THROWS_AS(std::bad_alloc, auto c = std::make_unique<int>(3));
    alloc_hook::disarm();
}

TEST(robustness, oom_id_map_add_all_or_nothing) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    // Sweep every add from 0 to 300 entries, so failures also hit the moments
    // when the internal arrays and hash table grow.
    IdMap m;
    bool all_completed = true;
    for (std::uint64_t i = 0; i < 300; ++i) {
        const long points = sweep_allocation_failures([&] { m.add(1000 + i); }, [&] {
            CHECK(m.size() == i && !m.find(1000 + i).has_value());
        });
        all_completed = all_completed && points > 0;
        if (m.size() != i + 1) break;
    }
    CHECK(all_completed);
    REQUIRE(m.size() == 300);
    bool consistent = true;
    for (std::uint64_t i = 0; i < 300; ++i) consistent = consistent && *m.find(1000 + i) == i && m.external(NodeId(i)) == 1000 + i;
    CHECK(consistent);
}

TEST(robustness, oom_vector_store_add_all_or_nothing) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    VectorStore s(8, 2);  // 4 rows per shelf: the 5th add needs a new shelf
    for (int i = 0; i < 4; ++i) s.add(std::vector<float>(8, float(i)));
    const long points = sweep_allocation_failures([&] { s.add(std::vector<float>(8, 9.0f)); }, [&] {
        CHECK(s.size() == 4 && s.get(3)[0] == 3.0f);
    });
    CHECK(points > 0);
    CHECK(s.size() == 5 && s.get(4)[7] == 9.0f && aligned(s.get(4).data()));
}

TEST(robustness, oom_graph_add_node_all_or_nothing) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    GraphStorage g(16, 2);
    for (int i = 0; i < 4; ++i) g.add_node(0);
    const long points = sweep_allocation_failures([&] { g.add_node(3); }, [&] {
        CHECK(g.size() == 4);
    });
    CHECK(points > 0);
    CHECK(g.size() == 5 && g.level(4) == 3);
    bool empty = true;
    for (int l = 0; l <= 3; ++l) empty = empty && GraphStorage::count(g.links(4, l)) == 0;
    CHECK(empty);
}

TEST(robustness, oom_storage_insert_all_or_nothing) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    for (std::size_t existing : std::initializer_list<std::size_t>{0, 1, 64}) {  // first insert allocates the most
        Storage st(8, 4);
        for (std::size_t i = 0; i < existing; ++i) st.insert(i, std::vector<float>(8, 1.0f), int(i % 3));
        const long points = sweep_allocation_failures(
            [&] { st.insert(777, std::vector<float>(8, 2.0f), 2); },
            [&] {
                CHECK(st.size() == existing && st.ids().size() == existing && st.graph().size() == existing);
                CHECK(!st.ids().find(777).has_value());
            });
        CHECK(points > 0);
        REQUIRE(st.size() == existing + 1);
        const NodeId id = *st.ids().find(777);
        CHECK(st.vectors().get(id)[0] == 2.0f && st.graph().level(id) == 2);
        CHECK(st.ids().size() == st.graph().size());
    }
}

TEST(robustness, oom_flat_add_all_or_nothing) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = random_vectors(51, 8, 83);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 50; ++i) f.add(i, data[i]);
    const Results before = f.search(data[0], 10);
    const long points = sweep_allocation_failures([&] { f.add(999, data[50]); }, [&] {
        CHECK(f.size() == 50 && !f.contains(999));
        CHECK(same_results(f.search(data[0], 10), before));
    });
    CHECK(points > 0);
    CHECK(f.size() == 51 && f.contains(999));  // the same ID works after failures
    CHECK(f.search(data[50], 1)[0].id == 999);
}

TEST(robustness, oom_flat_search_harmless) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = random_vectors(100, 8, 84);
    FlatIndex f(8, Metric::Cosine);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    const Results before = f.search(data[7], 10);
    Results got;
    const long points = sweep_allocation_failures([&] { got = f.search(data[7], 10); }, [&] {
        CHECK(same_results(f.search(data[7], 10), before));
    });
    CHECK(points > 0 && same_results(got, before));
}

TEST(robustness, oom_hnsw_first_insert) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    HnswIndex h(4, Metric::L2);
    const long points = sweep_allocation_failures([&] { h.add(1, std::vector<float>{1, 2, 3, 4}); }, [&] {
        CHECK(h.size() == 0 && h.entry_point() == kEmpty && h.storage().size() == 0);
    });
    CHECK(points > 0);
    CHECK(h.size() == 1 && h.contains(1) && h.entry_point() == 0);
}

TEST(robustness, oom_hnsw_add_keeps_index_consistent) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(400, 8, 4, 85);
    auto queries = clustered(5, 8, 4, 86);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) h.add(i, data[i]);
    const std::uint64_t last_id = 10000;  // a failed insert frees its ID, so retry the same one
    const long points = sweep_allocation_failures(
        [&] { h.add(last_id, data[300]); },
        [&] {
            CHECK(h.size() == 300);          // the failed vector is not counted...
            CHECK(!h.contains(last_id));     // ...and never visible
            check_graph(h, 0.99);
            for (const auto& q : queries) check_results(h, h.search(q, 10, 64));
        });
    CHECK(points > 0);
    CHECK(h.size() == 301 && h.contains(last_id));
    CHECK(h.search(data[300], 1, 64)[0].id == last_id);
    for (std::size_t i = 301; i < 400; ++i) h.add(i, data[i]);  // keeps working afterwards
    check_graph(h, 0.99);
}

TEST(robustness, oom_hnsw_search_harmless) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 8, 4, 87);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Results before = h.search(data[5], 10, 64);
    Results got;
    const long points = sweep_allocation_failures([&] { got = h.search(data[5], 10, 64); }, [&] {
        CHECK(same_results(h.search(data[5], 10, 64), before));
    });
    CHECK(points > 0 && same_results(got, before));
}

TEST(robustness, oom_index_construction) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    long p1 = sweep_allocation_failures([] { HnswIndex h(16, Metric::L2); h.add(1, std::vector<float>(16, 1.0f)); }, [] {});
    long p2 = sweep_allocation_failures([] { FlatIndex f(16, Metric::L2); f.add(1, std::vector<float>(16, 1.0f)); }, [] {});
    CHECK(p1 > 0 && p2 > 0);
}

TEST(robustness, oom_visited_pool) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    VisitedListPool pool;
    const long points = sweep_allocation_failures([&] { auto h = pool.acquire(1000); h->visit(999); }, [&] {
        CHECK(pool.idle_count() <= 1);
    });
    CHECK(points > 0 && pool.idle_count() == 1);
}

// ===========================================================================
// Real deletion (docs/core-capabilities-test-plan.md, scenarios D1-D66).
// Test names start with the scenario ID.
// ===========================================================================

namespace {

/// Points around `clusters` centers, also returning each point's cluster.
std::pair<Vectors, std::vector<std::size_t>> clustered_labeled(std::size_t count, std::size_t dim,
                                                               std::size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    Vectors centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (float& x : c) x = center(rng);
    Vectors out(count, std::vector<float>(dim));
    std::vector<std::size_t> label(count);
    for (std::size_t i = 0; i < count; ++i) {
        label[i] = i % clusters;
        for (std::size_t d = 0; d < dim; ++d) out[i][d] = centers[label[i]][d] + noise(rng);
    }
    return {out, label};
}

/// Exact top-k over a map of live vectors, sorted by distance then user ID.
Results brute_force(const std::map<std::uint64_t, std::vector<float>>& live,
                    const std::vector<float>& query, Metric m, std::size_t k) {
    const std::size_t dim = query.size();
    PreparedVector q(dim), row(dim);
    q.prepare(query, m);
    DistanceFn fn = get_distance(m);
    Results all;
    for (const auto& [id, v] : live) {
        row.prepare(v, m);
        all.push_back({id, fn(q.data(), row.data(), q.stride())});
    }
    all = tie_sorted(all);
    if (all.size() > k) all.resize(k);
    return all;
}

/// Removes and re-adds vectors one at a time (M = 4, so levels vary a lot) and
/// reports whether a reused slot ever got a higher, and a lower, level.
struct LevelChange { bool higher = false, lower = false, same_slot = true; };
LevelChange reuse_levels(HnswIndex& h, const Vectors& replacement) {
    LevelChange seen;
    std::uint64_t next = 100000;
    for (std::uint64_t i = 0; i < replacement.size() && !(seen.higher && seen.lower); ++i) {
        const NodeId slot = *h.storage().ids().find(i);
        const int old_level = h.storage().graph().level(slot);
        h.remove(i);
        h.add(next, replacement[i]);
        const NodeId reused = *h.storage().ids().find(next++);
        seen.same_slot = seen.same_slot && reused == slot;  // last in, first out
        const int new_level = h.storage().graph().level(reused);
        seen.higher = seen.higher || new_level > old_level;
        seen.lower = seen.lower || new_level < old_level;
    }
    return seen;
}

}  // namespace

// ---- Layer 1 building blocks ----------------------------------------------

TEST(deletion, d01_id_map_release_keeps_slot) {
    IdMap m;
    m.add(10);
    m.add(11);
    auto slot = m.release(10);
    CHECK(slot.has_value() && *slot == 0);
    CHECK(!m.find(10).has_value() && m.size() == 2);
    CHECK(m.is_deleted(0) && m.is_free(0));
    CHECK(*m.find(11) == 1 && !m.is_free(1));
}

TEST(deletion, d02_id_map_release_unknown) {
    IdMap m;
    m.add(1);
    CHECK(!m.release(99).has_value());
    CHECK(m.release(1).has_value());
    CHECK(!m.release(1).has_value());  // already released
    CHECK(m.size() == 1);
}

TEST(deletion, d03_id_map_bind_free_slot) {
    IdMap m;
    m.add(10);
    m.release(10);
    m.bind(0, 20);
    CHECK(*m.find(20) == 0 && m.external(0) == 20);
    CHECK(!m.is_deleted(0) && !m.is_free(0) && !m.find(10).has_value());
}

TEST(deletion, d04_id_map_bind_duplicate_rejected) {
    IdMap m;
    m.add(10);
    m.add(11);
    m.release(10);
    CHECK_THROWS_AS(std::invalid_argument, m.bind(0, 11));
    CHECK(m.is_free(0) && *m.find(11) == 1);
}

TEST(deletion, d05_id_map_bind_non_free_rejected) {
    IdMap m;
    m.add(10);
    CHECK_THROWS_AS(std::logic_error, m.bind(0, 20));   // live slot
    m.mark_deleted(0);                                   // tombstone: ID still mapped
    CHECK_THROWS_AS(std::logic_error, m.bind(0, 20));
    CHECK_THROWS_AS(std::logic_error, m.bind(5, 20));   // no such slot
    CHECK(!m.find(20).has_value() && *m.find(10) == 0);
}

TEST(deletion, d06_free_list_last_in_first_out) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 5; ++i) h.add(i, std::vector<float>{float(i), 0});
    h.remove(1);
    h.remove(3);
    h.add(100, std::vector<float>{7, 7});
    h.add(101, std::vector<float>{8, 8});
    h.add(102, std::vector<float>{9, 9});
    const IdMap& ids = h.storage().ids();
    CHECK(*ids.find(100) == 3);  // freed last, reused first
    CHECK(*ids.find(101) == 1);
    CHECK(*ids.find(102) == 5);  // free list empty: appended
}

TEST(deletion, d07_vector_store_overwrite) {
    VectorStore s(5);
    s.add(std::vector<float>{1, 2, 3, 4, 5});
    const float* address = s.get(0).data();
    s.overwrite(0, std::vector<float>{9, 9, 9, 9, 9});
    CHECK(s.get(0).data() == address && s.get(0)[0] == 9.0f && s.get(0)[4] == 9.0f);
    bool zero = true;
    for (std::size_t i = 5; i < s.stride(); ++i) zero = zero && s.get_padded(0)[i] == 0.0f;
    CHECK(zero);
}

TEST(deletion, d08_vector_store_overwrite_rejected) {
    VectorStore s(3);
    s.add(std::vector<float>{1, 2, 3});
    CHECK_THROWS_AS(std::invalid_argument, s.overwrite(0, std::vector<float>{9, 9}));
    CHECK_THROWS_AS(std::out_of_range, s.overwrite(1, std::vector<float>{9, 9, 9}));
    CHECK(s.get(0)[0] == 1.0f && s.get(0)[2] == 3.0f);
}

TEST(deletion, d09_vector_store_move_row) {
    VectorStore s(5);
    s.add(std::vector<float>{1, 1, 1, 1, 1});
    s.add(std::vector<float>{2, 3, 4, 5, 6});
    s.move_row(1, 0);
    CHECK(s.get(0)[0] == 2.0f && s.get(0)[4] == 6.0f);
    bool zero = true;
    for (std::size_t i = 5; i < s.stride(); ++i) zero = zero && s.get_padded(0)[i] == 0.0f;
    CHECK(zero);
    CHECK_THROWS_AS(std::out_of_range, s.move_row(5, 0));
    CHECK(s.get(0)[0] == 2.0f);
}

TEST(deletion, d10_vector_store_pop_back_reuses_row) {
    VectorStore s(4, 2);  // 4 rows per shelf
    for (int i = 0; i < 4; ++i) s.add(std::vector<float>(4, float(i)));
    const float* last = s.get(3).data();
    s.pop_back();
    CHECK(s.size() == 3);
    s.add(std::vector<float>(4, 7.0f));
    CHECK(s.size() == 4 && s.get(3).data() == last && s.get(3)[0] == 7.0f);  // same row, no new shelf
}

TEST(deletion, d11_graph_reset_same_level) {
    GraphStorage g(16);
    NodeId n = g.add_node(2);
    for (int l = 0; l <= 2; ++l) g.set_links(n, l, std::vector<NodeId>{1, 2, 3});
    g.reset_node(n, 2);
    bool empty = true;
    for (int l = 0; l <= 2; ++l) empty = empty && GraphStorage::count(g.links(n, l)) == 0;
    CHECK(empty && g.level(n) == 2);
}

TEST(deletion, d12_graph_reset_higher_level) {
    GraphStorage g(16);
    NodeId n = g.add_node(1);
    g.set_links(n, 0, std::vector<NodeId>{4, 5});
    g.set_links(n, 1, std::vector<NodeId>{6});
    g.reset_node(n, 3);
    CHECK(g.level(n) == 3 && g.links(n, 3).size() == 16);
    bool empty = true;
    for (int l = 0; l <= 3; ++l) empty = empty && GraphStorage::count(g.links(n, l)) == 0;
    CHECK(empty);
}

TEST(deletion, d13_graph_reset_lower_level) {
    GraphStorage g(16);
    NodeId n = g.add_node(3);
    g.reset_node(n, 1);
    CHECK(g.level(n) == 1);
    CHECK_THROWS_AS(std::out_of_range, g.links(n, 2));
    CHECK(GraphStorage::count(g.links(n, 1)) == 0);
}

TEST(deletion, d14_graph_reset_invalid) {
    GraphStorage g(16);
    NodeId n = g.add_node(1);
    g.set_links(n, 0, std::vector<NodeId>{7});
    CHECK_THROWS_AS(std::invalid_argument, g.reset_node(n, -1));
    CHECK_THROWS_AS(std::invalid_argument, g.reset_node(n, 256));
    CHECK_THROWS_AS(std::out_of_range, g.reset_node(99, 0));
    CHECK(g.level(n) == 1 && g.links(n, 0)[0] == 7);  // unchanged
}

TEST(deletion, d15_oom_building_blocks) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    IdMap m;
    for (std::uint64_t i = 0; i < 50; ++i) m.add(i);
    m.release(7);
    CHECK(sweep_allocation_failures([&] { m.bind(7, 5000); },
                                    [&] { CHECK(m.is_free(7) && !m.find(5000).has_value()); }) >= 0);
    CHECK(*m.find(5000) == 7);

    GraphStorage g(16);
    NodeId n = g.add_node(0);  // level 0: the arena has no block yet, so reset must allocate
    g.set_links(n, 0, std::vector<NodeId>{9});
    CHECK(sweep_allocation_failures([&] { g.reset_node(n, 4); },
                                    [&] { CHECK(g.level(n) == 0 && g.links(n, 0)[0] == 9); }) > 0);
    CHECK(g.level(n) == 4);

    Storage st(4);
    for (std::uint64_t i = 0; i < 3; ++i) st.insert(i, std::vector<float>(4, float(i)), 0);
    st.ids().release(1);
    CHECK(sweep_allocation_failures([&] { st.insert_into(1, 99, std::vector<float>(4, 9.0f), 3); }, [&] {
        CHECK(st.ids().is_free(1) && !st.ids().find(99).has_value() && st.size() == 3);
    }) > 0);
    CHECK(*st.ids().find(99) == 1 && st.graph().level(1) == 3 && st.vectors().get(1)[0] == 9.0f);
}

// ---- FlatIndex: swap-with-last removal -------------------------------------

TEST(deletion, d20_flat_remove_middle) {
    auto data = random_vectors(100, 8, 201);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(1000 + i, data[i]);
    std::vector<bool> removed(100, false);
    CHECK(f.remove(1050));
    removed[50] = true;
    for (const auto& q : random_vectors(5, 8, 202))
        CHECK(same_results(tie_sorted(f.search(q, 100)), tie_sorted(reference_search(data, removed, q, Metric::L2, 100, 1000))));
}

TEST(deletion, d21_flat_remove_last_moves_nothing) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    f.add(2, std::vector<float>{1, 1});
    f.add(3, std::vector<float>{2, 2});
    CHECK(f.remove(3));
    auto r = f.search(std::vector<float>{0, 0}, 5);
    CHECK(r.size() == 2 && r[0].id == 1 && r[1].id == 2 && f.capacity() == 2);
}

TEST(deletion, d22_flat_remove_only_then_add) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    CHECK(f.remove(1) && f.size() == 0 && f.capacity() == 0);
    CHECK(f.search(std::vector<float>{0, 0}, 3).empty());
    f.add(2, std::vector<float>{1, 1});
    CHECK(f.search(std::vector<float>{0, 0}, 3)[0].id == 2);
}

TEST(deletion, d23_flat_remove_all_random_then_readd) {
    auto data = random_vectors(300, 8, 203);
    FlatIndex f(8, Metric::L2), fresh(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    std::vector<std::uint64_t> order(300);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(204));
    for (auto id : order) CHECK(f.remove(id));
    CHECK(f.size() == 0 && f.capacity() == 0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        f.add(i, data[i]);
        fresh.add(i, data[i]);
    }
    bool same = true;
    for (const auto& q : random_vectors(10, 8, 205)) same = same && same_results(tie_sorted(f.search(q, 20)), tie_sorted(fresh.search(q, 20)));
    CHECK(same);
}

TEST(deletion, d24_flat_remove_moved_vector) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});   // slot 0
    f.add(2, std::vector<float>{5, 5});   // slot 1
    f.add(3, std::vector<float>{9, 9});   // slot 2
    CHECK(f.remove(1));                   // 3 moves into slot 0
    CHECK(f.remove(3));                   // remove the moved vector
    auto r = f.search(std::vector<float>{5, 5}, 5);
    CHECK(r.size() == 1 && r[0].id == 2 && r[0].distance == 0.0f);
}

TEST(deletion, d25_flat_reuse_id_many_times) {
    FlatIndex f(2, Metric::L2);
    f.add(99, std::vector<float>{100, 100});
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        f.add(7, std::vector<float>{float(i), 0});
        auto r = f.search(std::vector<float>{float(i), 0}, 1);
        ok = ok && r[0].id == 7 && r[0].distance == 0.0f;
        ok = ok && f.remove(7);
    }
    CHECK(ok && f.size() == 1);
}

TEST(deletion, d26_flat_remove_return_values) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    CHECK(!f.remove(2));
    CHECK(f.remove(1) && !f.contains(1));
    CHECK(!f.remove(1));
}

TEST(deletion, d27_flat_has_no_tombstones) {
    auto data = random_vectors(200, 4, 206);
    FlatIndex f(4, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    for (std::size_t i = 0; i < 200; i += 3) f.remove(i);
    CHECK(f.capacity() == f.size() && f.deleted_count() == 0);
}

TEST(deletion, d28_flat_churn_against_map) {
    auto pool = random_vectors(4000, 8, 207);
    std::mt19937 rng(208);
    FlatIndex f(8, Metric::L2);
    std::map<std::uint64_t, std::vector<float>> live;
    std::vector<std::uint64_t> ids;
    std::size_t next = 0;
    bool sizes = true, results = true;
    for (int step = 0; step < 10000; ++step) {
        if ((rng() % 2 == 0 || ids.empty()) && next < pool.size()) {
            f.add(next, pool[next]);
            live[next] = pool[next];
            ids.push_back(next++);
        } else if (!ids.empty()) {
            const std::size_t pos = rng() % ids.size();
            const std::uint64_t id = ids[pos];
            ids[pos] = ids.back();
            ids.pop_back();
            results = results && f.remove(id);
            live.erase(id);
        }
        sizes = sizes && f.size() == live.size() && f.capacity() == live.size();
        if (step % 50 == 0) {
            const auto& q = pool[rng() % pool.size()];
            results = results && same_results(tie_sorted(f.search(q, 10)), brute_force(live, q, Metric::L2, 10));
        }
    }
    CHECK(sizes);
    CHECK(results);
}

TEST(deletion, d29_flat_extreme_ids_reused) {
    FlatIndex f(2, Metric::L2);
    f.add(0, std::vector<float>{0, 0});
    f.add(kMaxId, std::vector<float>{1, 1});
    CHECK(f.remove(0) && f.remove(kMaxId));
    f.add(kMaxId, std::vector<float>{3, 3});
    f.add(0, std::vector<float>{4, 4});
    CHECK(f.search(std::vector<float>{3, 3}, 1)[0].id == kMaxId);
    CHECK(f.search(std::vector<float>{4, 4}, 1)[0].id == 0);
}

TEST(deletion, d30_flat_remove_never_allocates) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = random_vectors(50, 4, 209);
    FlatIndex f(4, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    const long points = sweep_allocation_failures([&] { f.remove(20); }, [&] { CHECK(f.contains(20)); });
    CHECK(points == 0);  // completed on the first try: no allocation at all
    CHECK(!f.contains(20) && f.size() == 49);
}

// ---- HnswIndex: free list, repair, slot reuse ------------------------------

TEST(deletion, d40_hnsw_reused_id_gets_new_vector) {
    auto data = clustered(300, 8, 4, 210);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    h.remove(5);
    h.add(5, std::vector<float>(8, 10.0f));  // far from everything
    bool old_gone = true;
    for (const auto& x : h.search(data[5], 10, 100)) old_gone = old_gone && x.id != 5;
    CHECK(old_gone);
    auto r = h.search(std::vector<float>(8, 10.0f), 1, 100);
    CHECK(r[0].id == 5 && r[0].distance == 0.0f);
}

TEST(deletion, d41_hnsw_insert_reuses_slot) {
    auto data = clustered(200, 8, 4, 211);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 50; ++i) h.remove(i);
    for (std::uint64_t i = 0; i < 50; ++i) h.add(1000 + i, data[i]);
    CHECK(h.capacity() == 200 && h.deleted_count() == 0 && h.size() == 200);
}

TEST(deletion, d42_hnsw_reused_slot_higher_level) {
    auto data = clustered(300, 8, 4, 212);
    HnswIndex h(8, Metric::L2, HnswParams{4, 50, 3});
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const LevelChange seen = reuse_levels(h, clustered(300, 8, 4, 213));
    CHECK(seen.higher && seen.same_slot);
    check_graph(h, 0.95);
    for (const auto& q : clustered(10, 8, 4, 214)) check_results(h, h.search(q, 10, 64));
}

TEST(deletion, d43_hnsw_reused_slot_lower_level) {
    auto data = clustered(300, 8, 4, 215);
    HnswIndex h(8, Metric::L2, HnswParams{4, 50, 5});
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const LevelChange seen = reuse_levels(h, clustered(300, 8, 4, 216));
    CHECK(seen.lower && seen.same_slot);
    check_graph(h, 0.95);
    for (const auto& q : clustered(10, 8, 4, 217)) check_results(h, h.search(q, 10, 64));
}

TEST(deletion, d44_hnsw_repair_unlinks_removed) {
    auto data = clustered(500, 8, 4, 218);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Storage& st = h.storage();
    const NodeId x = *st.ids().find(10);
    std::vector<std::pair<NodeId, int>> former;  // (neighbor, level)
    for (int l = 0; l <= st.graph().level(x); ++l)
        for (NodeId nb : st.graph().links(x, l)) {
            if (nb == kEmpty) break;
            former.push_back({nb, l});
        }
    REQUIRE(!former.empty());
    h.remove(10);
    bool unlinked = true;
    for (auto [nb, l] : former)
        for (NodeId c : st.graph().links(nb, l)) {
            if (c == kEmpty) break;
            unlinked = unlinked && c != x;
        }
    CHECK(unlinked);
    check_graph(h, 0.99);
}

TEST(deletion, d45_hnsw_reachable_after_removing_30_percent) {
    auto data = clustered(2000, 16, 10, 219);
    auto queries = clustered(40, 16, 10, 220);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    std::vector<std::uint64_t> order(2000);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(221));
    for (std::size_t i = 0; i < 600; ++i) {
        h.remove(order[i]);
        f.remove(order[i]);
    }
    check_graph(h, 0.99);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
}

TEST(deletion, d46_hnsw_repair_disabled) {
    auto data = clustered(2000, 16, 10, 219);
    auto queries = clustered(40, 16, 10, 220);
    HnswParams p;
    p.repair_on_remove = false;
    HnswIndex h(16, Metric::L2, p);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::uint64_t i = 0; i < 2000; i += 3) {
        h.remove(i);
        f.remove(i);
    }
    check_graph(h, 0.98);
    CHECK(recall(h, f, queries, 10, 100) >= 0.90);
}

TEST(deletion, d47_hnsw_remove_entry_point_repeatedly) {
    auto data = clustered(500, 8, 4, 222);
    auto queries = clustered(20, 8, 4, 223);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    bool valid = true;
    for (int i = 0; i < 20; ++i) {
        const std::uint64_t entry_id = h.storage().ids().external(h.entry_point());
        h.remove(entry_id);
        f.remove(entry_id);
        int highest = -1;
        for (NodeId n = 0; n < h.capacity(); ++n)
            if (!h.storage().ids().is_deleted(n)) highest = std::max(highest, h.storage().graph().level(n));
        valid = valid && !h.storage().ids().is_deleted(h.entry_point()) && h.max_level() == highest;
    }
    CHECK(valid);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
}

TEST(deletion, d48_hnsw_remove_everything_state) {
    auto data = clustered(100, 8, 4, 224);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 100; ++i) h.remove(i);
    CHECK(h.entry_point() == kEmpty && h.max_level() == -1);
    CHECK(h.size() == 0 && h.capacity() == 100 && h.deleted_count() == 100);
    CHECK(h.search(data[0], 5).empty());
    h.add(777, data[3]);
    const NodeId slot = *h.storage().ids().find(777);
    CHECK(h.capacity() == 100);  // reused a slot
    CHECK(h.entry_point() == slot && h.max_level() == h.storage().graph().level(slot));
    check_graph(h, 1.0);
}

TEST(deletion, d49_hnsw_remove_all_readd_same_ids) {
    auto data = clustered(800, 16, 8, 225);
    auto queries = clustered(30, 16, 8, 226);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::uint64_t i = 0; i < 800; ++i) h.remove(i);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    CHECK(h.capacity() == 800 && h.size() == 800);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    check_graph(h, 0.99);
}

namespace {

/// Churn: `cycles` rounds of removing 20% of live vectors and adding as many
/// new ones, checking recall, graph and memory after every round.
void churn(Metric m, std::size_t count, int cycles, double min_recall, double min_reach, unsigned seed) {
    const std::size_t dim = 16;
    auto pool = clustered(count * std::size_t(cycles), dim, 10, seed);
    auto queries = clustered(30, dim, 10, seed + 1);
    HnswIndex h(dim, m);
    FlatIndex f(dim, m);
    std::vector<std::uint64_t> live;
    std::size_t next = 0;
    for (; next < count; ++next) {
        h.add(next, pool[next]);
        f.add(next, pool[next]);
        live.push_back(next);
    }
    std::mt19937 rng(seed + 2);
    double worst = 1.0;
    bool memory_bounded = true;
    for (int c = 0; c < cycles; ++c) {
        std::shuffle(live.begin(), live.end(), rng);
        const std::size_t drop = count / 5;
        for (std::size_t i = 0; i < drop; ++i) {
            h.remove(live.back());
            f.remove(live.back());
            live.pop_back();
        }
        for (std::size_t i = 0; i < drop && next < pool.size(); ++i, ++next) {
            h.add(next, pool[next]);
            f.add(next, pool[next]);
            live.push_back(next);
        }
        memory_bounded = memory_bounded && h.capacity() <= count;
        worst = std::min(worst, recall(h, f, queries, 10, 100));
        check_graph(h, min_reach);
    }
    CHECK(memory_bounded);
    CHECK(worst >= min_recall);
}

}  // namespace

TEST(deletion, d50_hnsw_churn_l2) { churn(Metric::L2, 1000, 20, 0.95, 0.99, 227); }

TEST(deletion, d51_hnsw_remove_whole_cluster) {
    auto [data, label] = clustered_labeled(1200, 16, 8, 228);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::size_t i = 0; i < data.size(); ++i)
        if (label[i] == 3) {
            h.remove(i);
            f.remove(i);
        }
    Vectors queries;
    for (std::size_t i = 0; i < data.size() && queries.size() < 20; ++i)
        if (label[i] == 3) queries.push_back(data[i]);  // search where the cluster was
    for (const auto& q : queries)
        for (const auto& r : h.search(q, 10, 100)) CHECK(label[r.id] != 3);
    CHECK(recall(h, f, queries, 10, 200) >= 0.90);
    check_graph(h, 0.99);
}

TEST(deletion, d52_hnsw_churn_ip_and_cosine) {
    churn(Metric::Cosine, 800, 8, 0.95, 0.99, 229);
    churn(Metric::InnerProduct, 800, 8, 0.80, 0.95, 230);
}

TEST(deletion, d53_hnsw_remove_some_identical_copies) {
    HnswIndex h(8, Metric::L2);
    std::vector<float> same(8, 0.5f);
    auto noise = random_vectors(200, 8, 231);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i, same);
    for (std::uint64_t i = 0; i < 200; ++i) h.add(1000 + i, noise[i]);
    for (std::uint64_t i = 0; i < 100; i += 2) h.remove(i);  // remove half the copies
    // Not every copy stays reachable (fixed link slots), so check that copies are
    // found first, and that every result at distance 0 is a live (odd) copy.
    auto r = h.search(same, 10, 100);
    REQUIRE(!r.empty());
    CHECK(r[0].distance == 0.0f);
    bool live_copies = true;
    for (const auto& x : r)
        if (x.distance == 0.0f) live_copies = live_copies && x.id < 100 && x.id % 2 == 1;
    CHECK(live_copies);
    bool found = true;
    for (std::uint64_t i = 0; i < 200; ++i) found = found && h.search(noise[i], 1, 100)[0].id == 1000 + i;
    CHECK(found);
}

TEST(deletion, d54_hnsw_remove_with_minimal_M) {
    auto data = clustered(400, 8, 4, 232);
    HnswIndex h(8, Metric::L2, HnswParams{2, 100, 1});
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 400; i += 4) h.remove(i);
    check_graph(h, 0.70);
    for (const auto& q : clustered(10, 8, 4, 233)) check_results(h, h.search(q, 10, 64));
}

TEST(deletion, d55_hnsw_deletion_deterministic) {
    auto data = clustered(700, 8, 4, 234);
    HnswIndex a(8, Metric::L2), b(8, Metric::L2);
    for (HnswIndex* h : {&a, &b}) {
        for (std::size_t i = 0; i < 500; ++i) h->add(i, data[i]);
        for (std::uint64_t i = 0; i < 500; i += 3) h->remove(i);
        for (std::size_t i = 500; i < 700; ++i) h->add(i, data[i]);
    }
    CHECK(a.entry_point() == b.entry_point() && a.max_level() == b.max_level());
    bool same = true;
    for (std::size_t i = 0; i < 30; ++i) same = same && same_results(a.search(data[i], 10), b.search(data[i], 10));
    CHECK(same);
}

TEST(deletion, d56_oom_hnsw_remove) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 8, 4, 235);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const long points = sweep_allocation_failures([&] { h.remove(5); }, [&] {
        CHECK(h.contains(5) && h.size() == 300 && h.deleted_count() == 0);  // still stored...
        check_graph(h, 0.99);                                               // ...graph still valid
    });
    CHECK(points > 0);
    CHECK(!h.contains(5) && h.size() == 299 && h.deleted_count() == 1);
}

TEST(deletion, d57_oom_hnsw_insert_into_reused_slot) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(301, 8, 4, 236);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) h.add(i, data[i]);
    h.remove(5);
    const long points = sweep_allocation_failures([&] { h.add(9999, data[300]); }, [&] {
        CHECK(!h.contains(9999) && h.size() == 299 && h.deleted_count() == 1);  // slot still free
        check_graph(h, 0.99);
    });
    CHECK(points > 0);
    CHECK(h.contains(9999) && h.deleted_count() == 0 && h.capacity() == 300);
}

// ---- compact() and statistics -----------------------------------------------

TEST(deletion, d60_hnsw_compact_keeps_live) {
    auto data = clustered(1000, 16, 8, 237);
    auto queries = clustered(30, 16, 8, 238);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::uint64_t i = 0; i < 1000; i += 10) {
        for (std::uint64_t j = i; j < i + 3; ++j) {
            h.remove(j);
            f.remove(j);
        }
    }
    const CompactStats stats = h.compact();
    CHECK(stats.kept == 700 && stats.reclaimed == 300);
    bool ids_ok = true;
    for (std::uint64_t i = 0; i < 1000; ++i) ids_ok = ids_ok && h.contains(i) == (i % 10 >= 3);
    CHECK(ids_ok);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    check_graph(h, 0.99);
}

TEST(deletion, d61_hnsw_compact_is_dense) {
    auto data = clustered(400, 8, 4, 239);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 400; i += 2) h.remove(i);
    h.compact();
    CHECK(h.capacity() == 200 && h.size() == 200 && h.deleted_count() == 0);
    bool dense = true;
    for (std::uint64_t i = 1; i < 400; i += 2) dense = dense && *h.storage().ids().find(i) < 200;
    CHECK(dense);
}

TEST(deletion, d62_hnsw_compact_edge_cases) {
    HnswIndex empty(4, Metric::L2);
    CompactStats s = empty.compact();
    CHECK(s.kept == 0 && s.reclaimed == 0 && empty.entry_point() == kEmpty);

    // No deletions: same seed and insertion order rebuild exactly the same graph.
    auto data = clustered(300, 8, 4, 240);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    std::vector<Results> before;
    for (std::size_t i = 0; i < 20; ++i) before.push_back(h.search(data[i], 10));
    s = h.compact();
    CHECK(s.kept == 300 && s.reclaimed == 0);
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) same = same && same_results(before[i], h.search(data[i], 10));
    CHECK(same);

    // Everything deleted.
    for (std::uint64_t i = 0; i < 300; ++i) h.remove(i);
    s = h.compact();
    CHECK(s.kept == 0 && s.reclaimed == 300 && h.capacity() == 0);
    CHECK(h.entry_point() == kEmpty && h.max_level() == -1);
    h.add(1, data[0]);
    CHECK(h.search(data[0], 1)[0].id == 1);
}

TEST(deletion, d63_compact_statistics) {
    auto data = clustered(200, 8, 4, 241);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 50; ++i) h.remove(i);
    CompactStats first = h.compact();
    CompactStats second = h.compact();
    CHECK(first.kept == 150 && first.reclaimed == 50);
    CHECK(second.kept == 150 && second.reclaimed == 0);
}

TEST(deletion, d64_oom_hnsw_compact) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(200, 8, 4, 242);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 200; i += 4) h.remove(i);
    const Results before = h.search(data[1], 10);
    const long points = sweep_allocation_failures([&] { h.compact(); }, [&] {
        CHECK(h.size() == 150 && h.capacity() == 200);              // old index intact
        CHECK(same_results(h.search(data[1], 10), before));
    });
    CHECK(points > 0);
    CHECK(h.capacity() == 150 && h.deleted_count() == 0);
}

TEST(deletion, d65_flat_compact_is_noop) {
    auto data = random_vectors(10, 4, 243);
    FlatIndex f(4, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    for (std::uint64_t i = 0; i < 3; ++i) f.remove(i);
    const Results before = f.search(data[5], 7);
    const CompactStats s = f.compact();
    CHECK(s.kept == 7 && s.reclaimed == 0);
    CHECK(same_results(f.search(data[5], 7), before));
}

TEST(deletion, d66_statistics_consistent) {
    auto pool = clustered(1500, 8, 6, 244);
    std::mt19937 rng(245);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    std::vector<std::uint64_t> live;
    std::size_t next = 0, peak = 0;
    bool ok = true;
    for (int step = 0; step < 3000 && next < pool.size(); ++step) {
        if (rng() % 3 != 0 || live.empty()) {
            h.add(next, pool[next]);
            f.add(next, pool[next]);
            live.push_back(next++);
        } else {
            const std::size_t pos = rng() % live.size();
            h.remove(live[pos]);
            f.remove(live[pos]);
            live[pos] = live.back();
            live.pop_back();
        }
        peak = std::max(peak, live.size());
        ok = ok && h.size() == live.size() && f.size() == live.size();
        ok = ok && h.capacity() == h.size() + h.deleted_count() && h.capacity() <= peak;
        ok = ok && f.capacity() == f.size() && f.deleted_count() == 0;
    }
    CHECK(ok);
}

// ---- Added by the coverage audit (beyond the test plan) -------------------

namespace {

/// Portable uniform float in [-1, 1): built from the generator's raw bits,
/// because std::uniform_real_distribution differs between standard libraries.
float portable_uniform(std::mt19937& rng) {
    return float(rng() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

}  // namespace

TEST(deletion, insert_next_to_removed_nodes_stays_findable) {
    // Found by the coverage audit: in sparse graphs where nearly everything was
    // removed without repair, a search could start in a region that reaches no
    // live node and return nothing. Searches and inserts now retry from the
    // entry point. Sweeps small graphs (repair off, M 2-4) where this occurs.
    bool findable = true;
    for (unsigned seed = 1; seed <= 150; ++seed) {
        std::mt19937 rng(seed);
        HnswParams p;
        p.M = 2 + seed % 3;
        p.ef_construction = 1 + seed % 4;
        p.repair_on_remove = false;
        p.seed = seed;
        HnswIndex h(2, Metric::L2, p);
        const int n = 20 + int(seed % 30);
        for (int i = 0; i < n; ++i) h.add(std::uint64_t(i), std::vector<float>{portable_uniform(rng), portable_uniform(rng)});
        const std::uint64_t entry = h.storage().ids().external(h.entry_point());
        for (int i = 0; i < n; ++i)
            if (std::uint64_t(i) != entry) h.remove(std::uint64_t(i));
        for (std::uint64_t i = 0; i < 5; ++i) {
            std::vector<float> v{portable_uniform(rng), portable_uniform(rng)};
            h.add(1000 + i, v);
            auto r = h.search(v, 1, h.capacity());
            findable = findable && !r.empty() && r[0].id == 1000 + i && r[0].distance == 0.0f;
        }
        check_graph(h, 0.0, 1.0);  // structure only: tiny, mostly reused graphs have many stale links
    }
    CHECK(findable);
}

TEST(deletion, search_starting_from_removed_node) {
    // With repair off, the greedy descent can end on a removed node; the level-0
    // search must start there without ever returning it.
    auto data = clustered(600, 8, 4, 246);
    HnswParams p;
    p.repair_on_remove = false;
    HnswIndex h(8, Metric::L2, p);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    const Storage& st = h.storage();
    // Targets linked from the entry point on the top level: a search for one of
    // them moves there in its first greedy step, so it ends on a removed node.
    std::vector<std::uint64_t> targets;
    for (NodeId n : st.graph().links(h.entry_point(), h.max_level())) {
        if (n == kEmpty) break;
        targets.push_back(st.ids().external(n));
    }
    for (NodeId n = 0; n < st.size() && targets.size() < 10; ++n)  // plus other upper-level nodes
        if (st.graph().level(n) >= 1 && n != h.entry_point() &&
            std::find(targets.begin(), targets.end(), st.ids().external(n)) == targets.end())
            targets.push_back(st.ids().external(n));
    REQUIRE(!targets.empty());
    Vectors queries;
    for (std::uint64_t id : targets) {
        queries.push_back(data[id]);
        h.remove(id);
        f.remove(id);
    }
    bool never_returned = true;
    for (std::size_t i = 0; i < targets.size(); ++i)
        for (const auto& r : h.search(queries[i], 10, 100)) never_returned = never_returned && r.id != targets[i];
    CHECK(never_returned);
    CHECK(recall(h, f, queries, 10, 100) >= 0.90);
    h.add(9999, queries[0]);  // an insert that starts its search from removed nodes
    CHECK(h.search(queries[0], 1, 100)[0].id == 9999);
}

TEST(deletion, repair_skips_already_removed_links) {
    // Removing Y does not unlink X -> Y when Y does not link back to X. Removing
    // X afterwards must repair around it while ignoring the dead link to Y.
    auto data = clustered(1000, 8, 4, 247);
    HnswIndex h(8, Metric::L2, HnswParams{4, 100, 1});  // small lists fill up and get pruned
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Storage& st = h.storage();
    NodeId x = kEmpty, y = kEmpty;
    for (NodeId a = 0; a < st.size() && x == kEmpty; ++a)
        for (NodeId b : st.graph().links(a, 0)) {
            if (b == kEmpty) break;
            auto back = st.graph().links(b, 0);
            if (std::find(back.begin(), back.end(), a) == back.end()) {  // a -> b only
                x = a;
                y = b;
                break;
            }
        }
    REQUIRE(x != kEmpty);
    const std::uint64_t x_id = st.ids().external(x), y_id = st.ids().external(y);
    std::vector<NodeId> x_neighbors;
    for (NodeId nb : st.graph().links(x, 0)) {
        if (nb == kEmpty) break;
        if (nb != y) x_neighbors.push_back(nb);
    }
    CHECK(h.remove(y_id));
    auto still = st.graph().links(x, 0);
    CHECK(std::find(still.begin(), still.end(), y) != still.end());  // dead link remains
    CHECK(h.remove(x_id));
    bool clean = true;
    for (NodeId nb : x_neighbors)
        for (NodeId c : st.graph().links(nb, 0)) {
            if (c == kEmpty) break;
            clean = clean && c != x && c != y;  // neither removed node is linked
        }
    CHECK(clean);
    check_graph(h, 0.99);
}

TEST(deletion, id_map_move_released_slot) {
    // move_slot from a slot that was itself released keeps the destination free.
    IdMap m;
    m.add(1);
    m.add(2);
    m.add(3);
    m.release(1);  // slot 0
    m.release(3);  // slot 2, the last one
    m.move_slot(2, 0);
    m.pop_back_slot();
    CHECK(m.size() == 2 && m.is_free(0) && *m.find(2) == 1);
    CHECK(!m.find(1).has_value() && !m.find(3).has_value());

    // A released slot whose old user ID now belongs to another slot: moving it
    // must not touch that other slot's mapping.
    IdMap n;
    n.add(1);      // slot 0
    n.add(2);      // slot 1
    n.add(3);      // slot 2
    n.release(1);  // slot 0 free
    n.release(3);  // slot 2 free (its old ID is 3)
    n.bind(0, 3);  // ID 3 now lives in slot 0
    n.release(2);  // slot 1 free
    n.move_slot(2, 1);
    CHECK(*n.find(3) == 0 && n.is_free(1));
}

TEST(deletion, repair_skips_stale_links) {
    // A removed node's own list can hold a stale link: to a slot that was reused
    // at a lower level. Repair must ignore it. Built deliberately: find a one-way
    // upper-level link A -> S, reuse S until its level drops, then remove A.
    auto data = clustered(1000, 8, 4, 252);
    auto extra = clustered(200, 8, 4, 253);
    HnswIndex h(8, Metric::L2, HnswParams{4, 100, 1});
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Storage& st = h.storage();
    NodeId a = kEmpty, s = kEmpty;
    int level = 0;
    for (NodeId x = 0; x < st.size() && a == kEmpty; ++x)
        for (int l = 1; l <= st.graph().level(x) && a == kEmpty; ++l)
            for (NodeId y : st.graph().links(x, l)) {
                if (y == kEmpty) break;
                auto back = st.graph().links(y, l);
                if (std::find(back.begin(), back.end(), x) == back.end()) {  // x -> y only
                    a = x;
                    s = y;
                    level = l;
                    break;
                }
            }
    REQUIRE(a != kEmpty);
    std::uint64_t s_id = st.ids().external(s);
    for (std::size_t i = 0; i < extra.size() && st.graph().level(s) >= level; ++i) {
        h.remove(s_id);           // frees slot s (a still links to it: one-way)
        s_id = 50000 + i;
        h.add(s_id, extra[i]);    // last in, first out: reuses slot s with a new level
        REQUIRE(*st.ids().find(s_id) == s);
    }
    REQUIRE(st.graph().level(s) < level);  // a's level-`level` link to s is now stale
    CHECK(h.remove(st.ids().external(a)));
    check_graph(h, 0.95);
    for (const auto& q : clustered(10, 8, 4, 254)) check_results(h, h.search(q, 10, 64));
}

TEST(deletion, compact_cosine_index) {
    auto data = clustered(500, 16, 6, 248);
    auto queries = clustered(20, 16, 6, 249);
    HnswIndex h(16, Metric::Cosine);
    FlatIndex f(16, Metric::Cosine);
    fill(h, f, data);
    for (std::uint64_t i = 0; i < 500; i += 2) {
        h.remove(i);
        f.remove(i);
    }
    h.compact();  // stored vectors are already normalized; normalizing again is harmless
    CHECK(h.capacity() == 250 && h.size() == 250);
    CHECK(recall(h, f, queries, 10, 100) >= 0.95);
    bool self = true;
    for (std::size_t i = 1; i < 500; i += 50) self = self && h.search(data[i], 1, 64)[0].id == i;
    CHECK(self);
}

TEST(deletion, add_after_compact_appends) {
    auto data = clustered(110, 8, 4, 250);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 100; ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 20; ++i) h.remove(i);
    h.compact();
    CHECK(h.capacity() == 80 && h.deleted_count() == 0);
    for (std::size_t i = 100; i < 110; ++i) h.add(i, data[i]);  // no free slots: appended
    CHECK(h.capacity() == 90 && h.size() == 90);
    check_graph(h, 0.99);
}

TEST(deletion, remove_high_level_node_keeps_top_level) {
    // Removing an upper-level node that is not the entry point leaves the
    // entry point and the top level unchanged.
    auto data = clustered(800, 8, 4, 251);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const NodeId entry = h.entry_point();
    const int top = h.max_level();
    const Storage& st = h.storage();
    std::uint64_t target = kMaxId;
    for (NodeId n = 0; n < st.size() && target == kMaxId; ++n)
        if (n != entry && st.graph().level(n) >= 1) target = st.ids().external(n);
    REQUIRE(target != kMaxId);
    CHECK(h.remove(target));
    CHECK(h.entry_point() == entry && h.max_level() == top);
    check_graph(h, 0.99);
}

// ===========================================================================
// Concurrency
// ===========================================================================

namespace {

/// Runs `threads` threads that each search `index` for every query `rounds`
/// times; returns how many answers differed from `expected`.
template <class Search>
int parallel_mismatches(const Vectors& queries, const std::vector<Results>& expected,
                        int threads, int rounds, Search search) {
    std::atomic<int> mismatches{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (int r = 0; r < rounds; ++r)
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    const std::size_t q = (i + std::size_t(t) * 13) % queries.size();
                    if (!same_results(search(queries[q]), expected[q])) ++mismatches;
                }
        });
    for (auto& th : pool) th.join();
    return mismatches.load();
}

}  // namespace

TEST(concurrency, hnsw_parallel_searches) {
    const Fixture& fx = fixture(Metric::L2);
    std::vector<Results> expected;
    for (const auto& q : fx.queries) expected.push_back(fx.hnsw->search(q, 10, 64));
    CHECK(parallel_mismatches(fx.queries, expected, 8, 3,
                              [&](const std::vector<float>& q) { return fx.hnsw->search(q, 10, 64); }) == 0);
}

TEST(concurrency, flat_parallel_searches) {
    const Fixture& fx = fixture(Metric::L2);
    std::vector<Results> expected;
    for (const auto& q : fx.queries) expected.push_back(fx.flat->search(q, 10));
    CHECK(parallel_mismatches(fx.queries, expected, 8, 2,
                              [&](const std::vector<float>& q) { return fx.flat->search(q, 10); }) == 0);
}

TEST(concurrency, cosine_parallel_searches) {
    const Fixture& fx = fixture(Metric::Cosine);
    std::vector<Results> expected;
    for (const auto& q : fx.queries) expected.push_back(fx.hnsw->search(q, 10, 64));
    CHECK(parallel_mismatches(fx.queries, expected, 4, 3,
                              [&](const std::vector<float>& q) { return fx.hnsw->search(q, 10, 64); }) == 0);
}

TEST(concurrency, mixed_indexes_and_ef) {
    // Different threads use different indexes and ef values at the same time.
    const Fixture& l2 = fixture(Metric::L2);
    const Fixture& ip = fixture(Metric::InnerProduct);
    std::vector<Results> e1, e2, e3;
    for (const auto& q : l2.queries) {
        e1.push_back(l2.hnsw->search(q, 5, 20));
        e2.push_back(l2.hnsw->search(q, 20, 200));
        e3.push_back(ip.hnsw->search(q, 10, 64));
    }
    std::atomic<int> bad{0};
    std::thread t1([&] { bad += parallel_mismatches(l2.queries, e1, 2, 2, [&](const std::vector<float>& q) { return l2.hnsw->search(q, 5, 20); }); });
    std::thread t2([&] { bad += parallel_mismatches(l2.queries, e2, 2, 2, [&](const std::vector<float>& q) { return l2.hnsw->search(q, 20, 200); }); });
    std::thread t3([&] { bad += parallel_mismatches(l2.queries, e3, 2, 2, [&](const std::vector<float>& q) { return ip.hnsw->search(q, 10, 64); }); });
    t1.join();
    t2.join();
    t3.join();
    CHECK(bad.load() == 0);
}

// ===========================================================================
// Search features: shared helpers
// ===========================================================================

namespace {

const char* const kCategories[] = {"news", "sports", "tech", "music", "food"};

/// The five-field schema used by most search-feature tests.
std::vector<FieldSpec> search_fields() {
    return {{"category", FieldType::Keyword}, {"year", FieldType::Int}, {"price", FieldType::Float},
            {"in_stock", FieldType::Bool}, {"tags", FieldType::Tags}};
}

/// Deterministic metadata for vector `id`; every 11th vector has only a category.
Metadata meta_for(std::uint64_t id) {
    if (id % 11 == 0) return Metadata().set("category", kCategories[id % 5]);
    std::vector<std::string> tags;
    if (id % 2) tags.push_back("a");
    if (id % 3 == 0) tags.push_back("b");
    if (id % 7 == 0) tags.push_back("c");
    return Metadata()
        .set("category", kCategories[id % 5])
        .set("year", std::int64_t(2000 + id % 25))
        .set("price", double(id % 100) * 1.5)
        .set("in_stock", id % 3 == 0)
        .set_tags("tags", tags);
}

/// Independent reference for the filter rules (missing field: comparisons are
/// false; ! negates; has_all_tags with no tags is true when the field exists).
bool ref_eval(const Filter::Node& n, const Metadata& md) {
    using K = Filter::Node::Kind;
    switch (n.kind) {
        case K::All: return true;
        case K::And: return ref_eval(*n.children[0], md) && ref_eval(*n.children[1], md);
        case K::Or: return ref_eval(*n.children[0], md) || ref_eval(*n.children[1], md);
        case K::Not: return !ref_eval(*n.children[0], md);
        default: break;
    }
    const auto v = md.get(n.field);
    if (!v) return false;
    auto num = [](const Value& x) { return x.type == FieldType::Int ? double(x.i) : x.f; };
    auto equal = [&](const Value& x, const Value& lit) {
        if (x.type == FieldType::Int || x.type == FieldType::Float) return num(x) == num(lit);
        if (x.type == FieldType::Keyword) return x.s == lit.s;
        return x.b == lit.b;
    };
    auto has = [&](const std::string& t) { return std::find(v->tags.begin(), v->tags.end(), t) != v->tags.end(); };
    switch (n.kind) {
        case K::Exists: return true;
        case K::Compare:
            switch (n.op) {
                case CompareOp::Eq: return equal(*v, n.a);
                case CompareOp::Ne: return !equal(*v, n.a);
                case CompareOp::Lt: return num(*v) < num(n.a);
                case CompareOp::Le: return num(*v) <= num(n.a);
                case CompareOp::Gt: return num(*v) > num(n.a);
                default: return num(*v) >= num(n.a);
            }
        case K::Between: return num(*v) >= num(n.a) && num(*v) <= num(n.b);
        case K::In:
            for (const Value& x : n.values)
                if (equal(*v, x)) return true;
            return false;
        case K::HasTag: return has(n.tags[0]);
        case K::HasAnyTag:
            for (const auto& t : n.tags)
                if (has(t)) return true;
            return false;
        case K::HasAllTags:
            for (const auto& t : n.tags)
                if (!has(t)) return false;
            return true;
        default: return false;
    }
}

/// A random, valid filter over the five search fields, up to `depth` levels deep.
Filter random_filter(std::mt19937& rng, int depth) {
    const auto pick = static_cast<unsigned>(rng() % (depth > 0 ? 12 : 9));
    const int year = 2000 + int(rng() % 25);
    switch (pick) {
        case 0: return Filter::eq("category", kCategories[rng() % 5]);
        case 1: return Filter::ne("category", kCategories[rng() % 5]);
        case 2: return Filter::ge("year", year);
        case 3: return Filter::lt("price", double(rng() % 150));
        case 4: return Filter::between("year", year, year + int(rng() % 10));
        case 5: return Filter::eq("in_stock", rng() % 2 == 0);
        case 6: return Filter::has_tag("tags", rng() % 2 ? "a" : "c");
        case 7: return Filter::in("category", {kCategories[rng() % 5], kCategories[rng() % 5]});
        case 8: return Filter::exists("year");
        case 9: return random_filter(rng, depth - 1) && random_filter(rng, depth - 1);
        case 10: return random_filter(rng, depth - 1) || random_filter(rng, depth - 1);
        default: return !random_filter(rng, depth - 1);
    }
}

/// Search options with just a filter.
SearchOptions with_filter(Filter f, Strategy s = Strategy::Auto) {
    SearchOptions o;
    o.filter = std::move(f);
    o.strategy = s;
    return o;
}

/// Flattens queries into one buffer for batch search.
std::vector<float> flatten(const Vectors& queries) {
    std::vector<float> out;
    for (const auto& q : queries) out.insert(out.end(), q.begin(), q.end());
    return out;
}

/// Recall of filtered HNSW results against filtered Flat results.
double filtered_recall(const HnswIndex& h, const FlatIndex& f, const Vectors& queries, std::size_t k,
                       std::size_t ef, const SearchOptions& options) {
    double total = 0.0;
    std::size_t counted = 0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : f.search(q, k, options)) truth.insert(r.id);
        if (truth.empty()) continue;
        std::size_t hits = 0;
        for (const auto& r : h.search(q, k, ef, options)) hits += truth.count(r.id);
        total += double(hits) / double(truth.size());
        ++counted;
    }
    return counted ? total / double(counted) : 1.0;
}

/// A shared HNSW + Flat pair with metadata (3000 x 16), plus a "bucket" field
/// (id % 100) for filters of known selectivity. Built once, then reused.
struct MetaFixture {
    Vectors data, queries;
    std::unique_ptr<HnswIndex> hnsw;
    std::unique_ptr<FlatIndex> flat;
};
const MetaFixture& meta_fixture() {
    static std::unique_ptr<MetaFixture> fx;
    if (!fx) {
        fx = std::make_unique<MetaFixture>();
        fx->data = clustered(3000, 16, 12, 301);
        fx->queries = clustered(40, 16, 12, 302);
        auto fields = search_fields();
        fields.push_back({"bucket", FieldType::Int});
        fx->hnsw = std::make_unique<HnswIndex>(16, Metric::L2, HnswParams{}, Schema::strict(fields));
        fx->flat = std::make_unique<FlatIndex>(16, Metric::L2, Schema::strict(fields));
        for (std::size_t i = 0; i < fx->data.size(); ++i) {
            Metadata md = meta_for(i).set("bucket", std::int64_t(i % 100));
            fx->hnsw->add(i, fx->data[i], md);
            fx->flat->add(i, fx->data[i], md);
        }
    }
    return *fx;
}

}  // namespace

// ===========================================================================
// Metadata (MD1-MD52)
// ===========================================================================

TEST(metadata, md01_all_types_roundtrip) {
    FlatIndex f(2, Metric::L2, Schema::strict(search_fields()));
    Metadata md = Metadata().set("category", "news").set("year", 2024).set("price", 9.5)
                      .set("in_stock", true).set_tags("tags", {"ai", "chips"});
    f.add(1, std::vector<float>{0, 0}, md);
    auto got = f.get_metadata(1);
    REQUIRE(got.has_value());
    CHECK(*got == md);
}

TEST(metadata, md02_missing_fields_have_no_value) {
    HnswIndex h(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    h.add(1, std::vector<float>{0, 0}, Metadata().set("year", 1999));
    auto got = h.get_metadata(1);
    REQUIRE(got.has_value());
    CHECK(got->size() == 1 && got->get("year")->i == 1999 && !got->get("category").has_value());
}

TEST(metadata, md03_strict_unknown_field_rejected) {
    FlatIndex f(2, Metric::L2, Schema::strict(search_fields()));
    HnswIndex h(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    CHECK_THROWS_AS(std::invalid_argument, f.add(1, std::vector<float>{0, 0}, Metadata().set("colour", "red")));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, std::vector<float>{0, 0}, Metadata().set("colour", "red")));
    CHECK(f.size() == 0 && h.size() == 0 && h.capacity() == 0 && !f.contains(1) && !h.contains(1));
}

TEST(metadata, md04_wrong_type_rejected) {
    HnswIndex h(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    std::vector<float> v{0, 0};
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, v, Metadata().set("year", "2024")));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, v, Metadata().set("tags", 5)));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, v, Metadata().set("in_stock", 1)));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, v, Metadata().set("category", true)));
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, v, Metadata().set("year", 1.5)));
    CHECK(h.size() == 0);
    h.add(1, v, Metadata().set("price", 10));  // an integer is accepted for a float field
    auto p = h.get_metadata(1)->get("price");
    CHECK(p->type == FieldType::Float && p->f == 10.0);
}

TEST(metadata, md05_invalid_schemas_rejected) {
    CHECK_THROWS_AS(std::invalid_argument, Schema::strict({{"a", FieldType::Int}, {"a", FieldType::Float}}));
    CHECK_THROWS_AS(std::invalid_argument, Schema::strict({{"", FieldType::Int}}));
    CHECK_THROWS_AS(std::invalid_argument, Schema::dynamic({{"x", FieldType::Bool}, {"x", FieldType::Bool}}));
}

TEST(metadata, md06_empty_strict_schema) {
    FlatIndex f(2, Metric::L2, Schema::strict({}));
    CHECK_NOTHROW(f.add(1, std::vector<float>{0, 0}));
    CHECK_THROWS_AS(std::invalid_argument, f.add(2, std::vector<float>{0, 0}, Metadata().set("a", 1)));
    CHECK(f.size() == 1 && f.get_metadata(1)->empty());
}

TEST(metadata, md10_dynamic_field_created_on_first_use) {
    FlatIndex f(2, Metric::L2);
    CHECK(f.metadata().field_count() == 0);
    f.add(1, std::vector<float>{0, 0}, Metadata().set("colour", "red"));
    auto field = f.metadata().find_field("colour");
    REQUIRE(field.has_value());
    CHECK(f.metadata().field_type(*field) == FieldType::Keyword);
}

TEST(metadata, md11_dynamic_type_is_fixed) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0}, Metadata().set("colour", "red"));
    CHECK_THROWS_AS(std::invalid_argument, h.add(2, std::vector<float>{1, 1}, Metadata().set("colour", 5)));
    CHECK(h.size() == 1 && !h.contains(2));
}

TEST(metadata, md12_partial_schema_plus_dynamic) {
    FlatIndex f(2, Metric::L2, Schema::dynamic({{"year", FieldType::Int}}));
    CHECK_THROWS_AS(std::invalid_argument, f.add(1, std::vector<float>{0, 0}, Metadata().set("year", "x")));
    CHECK_NOTHROW(f.add(1, std::vector<float>{0, 0}, Metadata().set("year", 2020).set("note", "hi")));
    CHECK(f.metadata().field_count() == 2);
}

TEST(metadata, md13_failed_insert_creates_no_field) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    CHECK_THROWS_AS(std::invalid_argument, h.add(1, std::vector<float>{1, 1}, Metadata().set("zzz", 1)));
    CHECK_THROWS_AS(std::invalid_argument, h.add(2, std::vector<float>{kNaN, 1}, Metadata().set("yyy", 1)));
    CHECK(!h.metadata().find_field("zzz").has_value() && !h.metadata().find_field("yyy").has_value());
    CHECK(h.metadata().field_count() == 0);
}

TEST(metadata, md14_unused_fields_cost_nothing) {
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 1000; ++i) f.add(i, std::vector<float>{float(i), 0});
    CHECK(f.metadata().field_count() == 0 && f.metadata().dictionary_size() == 0);
    f.add(5000, std::vector<float>{0, 0}, Metadata().set("late", 1));
    CHECK(f.metadata().field_count() == 1);
}

TEST(metadata, md20_integer_extremes) {
    FlatIndex f(1, Metric::L2);
    const std::int64_t lo = std::numeric_limits<std::int64_t>::min(), hi = std::numeric_limits<std::int64_t>::max();
    f.add(1, std::vector<float>{0}, Metadata().set("n", lo));
    f.add(2, std::vector<float>{1}, Metadata().set("n", hi));
    f.add(3, std::vector<float>{2}, Metadata().set("n", std::int64_t(0)));
    f.add(4, std::vector<float>{3}, Metadata().set("n", std::int64_t(-7)));
    CHECK(f.get_metadata(1)->get("n")->i == lo && f.get_metadata(2)->get("n")->i == hi);
    CHECK(f.get_metadata(3)->get("n")->i == 0 && f.get_metadata(4)->get("n")->i == -7);
}

TEST(metadata, md21_float_nan_and_infinity_rejected) {
    HnswIndex h(1, Metric::L2);
    h.add(1, std::vector<float>{0}, Metadata().set("price", 1.0));
    for (double bad : {double(kNaN), double(kInf), -double(kInf)}) {
        CHECK_THROWS_AS(std::invalid_argument, h.add(2, std::vector<float>{1}, Metadata().set("price", bad)));
        CHECK_THROWS_AS(std::invalid_argument, h.set_metadata(1, Metadata().set("price", bad)));
    }
    CHECK(h.size() == 1 && h.get_metadata(1)->get("price")->f == 1.0);
}

TEST(metadata, md22_strings) {
    FlatIndex f(1, Metric::L2);
    const std::string big(1 << 20, 'x'), utf8 = "na\xc3\xafve \xe2\x98\x95";
    f.add(1, std::vector<float>{0}, Metadata().set("s", ""));
    f.add(2, std::vector<float>{1}, Metadata().set("s", big));
    f.add(3, std::vector<float>{2}, Metadata().set("s", utf8));
    f.add(4, std::vector<float>{3}, Metadata().set("s", "News"));
    f.add(5, std::vector<float>{4}, Metadata().set("s", "news"));
    CHECK(f.get_metadata(1)->get("s")->s.empty() && f.get_metadata(2)->get("s")->s == big);
    CHECK(f.get_metadata(3)->get("s")->s == utf8);
    auto r = f.search(std::vector<float>{3}, 5, with_filter(Filter::eq("s", "news")));
    CHECK(r.size() == 1 && r[0].id == 5);  // case-sensitive
}

TEST(metadata, md23_tag_sets) {
    FlatIndex f(1, Metric::L2);
    std::vector<std::string> many;
    for (int i = 0; i < 1000; ++i) many.push_back("t" + std::to_string(i));
    f.add(1, std::vector<float>{0}, Metadata().set_tags("tags", {}));
    f.add(2, std::vector<float>{1}, Metadata().set_tags("tags", {"x", "y", "x", "x"}));
    f.add(3, std::vector<float>{2}, Metadata().set_tags("tags", many));
    CHECK(f.get_metadata(1)->get("tags")->tags.empty());
    CHECK(f.get_metadata(2)->get("tags")->tags.size() == 2);  // repeats collapse
    CHECK(f.get_metadata(3)->get("tags")->tags.size() == 1000);
    CHECK(f.search(std::vector<float>{0}, 5, with_filter(Filter::exists("tags"))).size() == 3);
    CHECK(f.search(std::vector<float>{0}, 5, with_filter(Filter::has_any_tag("tags", {"x", "t999"}))).size() == 2);
}

TEST(metadata, md24_booleans) {
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("b", true));
    f.add(2, std::vector<float>{1}, Metadata().set("b", false));
    CHECK(f.get_metadata(1)->get("b")->b && !f.get_metadata(2)->get("b")->b);
    auto r = f.search(std::vector<float>{0}, 5, with_filter(Filter::eq("b", false)));
    CHECK(r.size() == 1 && r[0].id == 2);
}

TEST(metadata, md25_strings_stored_once) {
    FlatIndex f(1, Metric::L2);
    for (std::uint64_t i = 0; i < 1000; ++i) f.add(i, std::vector<float>{float(i)}, Metadata().set("category", kCategories[i % 5]));
    CHECK(f.metadata().dictionary_size() == 5);
}

TEST(metadata, md30_set_metadata_changes_only_given_fields) {
    auto data = clustered(300, 8, 4, 303);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    const Results before = h.search(data[3], 10);
    CHECK(h.set_metadata(1, Metadata().set("category", "food").set("year", 1990)));
    auto md = *h.get_metadata(1);
    CHECK(md.get("category")->s == "food" && md.get("year")->i == 1990);
    CHECK(md.get("tags") == meta_for(1).get("tags"));  // untouched
    CHECK(same_results(h.search(data[3], 10), before));  // vector and graph untouched
}

TEST(metadata, md31_unset_clears_a_field) {
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("a", 1).set("b", 2));
    CHECK(f.set_metadata(1, Metadata().unset("a")));
    auto md = *f.get_metadata(1);
    CHECK(!md.get("a").has_value() && md.get("b")->i == 2);
}

TEST(metadata, md32_set_metadata_unknown_or_removed) {
    HnswIndex h(1, Metric::L2);
    h.add(1, std::vector<float>{0});
    h.remove(1);
    CHECK(!h.set_metadata(1, Metadata().set("a", 1)));
    CHECK(!h.set_metadata(99, Metadata().set("a", 1)));
}

TEST(metadata, md33_set_metadata_invalid_unchanged) {
    FlatIndex f(1, Metric::L2, Schema::strict({{"a", FieldType::Int}}));
    f.add(1, std::vector<float>{0}, Metadata().set("a", 5));
    CHECK_THROWS_AS(std::invalid_argument, f.set_metadata(1, Metadata().set("a", "x")));
    CHECK_THROWS_AS(std::invalid_argument, f.set_metadata(1, Metadata().set("zz", 1)));
    CHECK(f.get_metadata(1)->get("a")->i == 5);
}

TEST(metadata, md34_get_metadata_unknown_or_removed) {
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("a", 1));
    f.remove(1);
    CHECK(!f.get_metadata(1).has_value() && !f.get_metadata(42).has_value());
}

TEST(metadata, md40_flat_swap_moves_metadata) {
    FlatIndex f(1, Metric::L2);
    for (std::uint64_t i = 0; i < 5; ++i) f.add(i, std::vector<float>{float(i)}, Metadata().set("n", std::int64_t(i * 10)));
    f.remove(1);  // vector 4 moves into slot 1
    for (std::uint64_t i : std::initializer_list<std::uint64_t>{0, 2, 3, 4}) CHECK(f.get_metadata(i)->get("n")->i == std::int64_t(i * 10));
    auto r = f.search(std::vector<float>{4}, 5, with_filter(Filter::eq("n", 40)));
    CHECK(r.size() == 1 && r[0].id == 4);
}

TEST(metadata, md41_reused_slot_does_not_inherit) {
    HnswIndex h(1, Metric::L2);
    h.add(1, std::vector<float>{0}, Metadata().set("secret", "yes"));
    h.add(2, std::vector<float>{5});
    h.remove(1);
    h.add(3, std::vector<float>{1});  // reuses slot 0
    CHECK(*h.storage().ids().find(3) == 0);
    CHECK(h.get_metadata(3)->empty());
    CHECK(h.search(std::vector<float>{0}, 5, 64, with_filter(Filter::exists("secret"))).empty());
}

TEST(metadata, md42_reuse_with_partial_metadata) {
    HnswIndex h(1, Metric::L2);
    h.add(1, std::vector<float>{0}, Metadata().set("a", 1).set("b", 2));
    h.add(2, std::vector<float>{5});
    h.remove(1);
    h.add(3, std::vector<float>{1}, Metadata().set("b", 9));
    auto md = *h.get_metadata(3);
    CHECK(md.size() == 1 && md.get("b")->i == 9);
}

TEST(metadata, md43_compact_keeps_metadata) {
    auto data = clustered(200, 8, 4, 304);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 0; i < 200; i += 3) h.remove(i);
    h.compact();
    bool same = true;
    for (std::uint64_t i = 0; i < 200; ++i)
        same = same && (i % 3 == 0 ? !h.get_metadata(i).has_value() : *h.get_metadata(i) == meta_for(i));
    CHECK(same);
    CHECK(h.metadata().mode() == SchemaMode::Strict);
}

TEST(metadata, md44_churn_against_reference) {
    for (int which = 0; which < 2; ++which) {
        std::unique_ptr<HnswIndex> h;
        std::unique_ptr<FlatIndex> f;
        if (which == 0) h = std::make_unique<HnswIndex>(4, Metric::L2);
        else f = std::make_unique<FlatIndex>(4, Metric::L2);
        auto pool = random_vectors(1500, 4, 305);
        std::map<std::uint64_t, Metadata> ref;
        std::vector<std::uint64_t> live;
        std::mt19937 rng(306);
        std::size_t next = 0;
        bool ok = true;
        for (int step = 0; step < 3000 && next < pool.size(); ++step) {
            const unsigned op = static_cast<unsigned>(rng() % 4);
            if (op < 2 || live.empty()) {
                Metadata md = meta_for(next);
                h ? h->add(next, pool[next], md) : f->add(next, pool[next], md);
                ref[next] = md;
                live.push_back(next++);
            } else if (op == 2) {
                const std::size_t pos = rng() % live.size();
                h ? h->remove(live[pos]) : f->remove(live[pos]);
                ref.erase(live[pos]);
                live[pos] = live.back();
                live.pop_back();
            } else {
                const std::uint64_t id = live[rng() % live.size()];
                Metadata change = Metadata().set("year", std::int64_t(rng() % 3000));
                h ? h->set_metadata(id, change) : f->set_metadata(id, change);
                ref[id].set("year", *change.get("year"));
            }
            if (step % 25 == 0)
                for (const auto& [id, md] : ref) ok = ok && *(h ? h->get_metadata(id) : f->get_metadata(id)) == md;
        }
        for (const auto& [id, md] : ref) ok = ok && *(h ? h->get_metadata(id) : f->get_metadata(id)) == md;
        CHECK(ok);
    }
}

TEST(metadata, md45_removed_vector_never_matches) {
    FlatIndex f(1, Metric::L2);
    HnswIndex h(1, Metric::L2);
    for (std::uint64_t i = 0; i < 20; ++i) {
        f.add(i, std::vector<float>{float(i)}, Metadata().set("k", i == 5 ? "only" : "other"));
        h.add(i, std::vector<float>{float(i)}, Metadata().set("k", i == 5 ? "only" : "other"));
    }
    f.remove(5);
    h.remove(5);
    CHECK(f.search(std::vector<float>{5}, 10, with_filter(Filter::eq("k", "only"))).empty());
    CHECK(h.search(std::vector<float>{5}, 10, 64, with_filter(Filter::eq("k", "only"))).empty());
    CHECK(!f.get_metadata(5).has_value() && !h.get_metadata(5).has_value());
}

TEST(metadata, md50_oom_add_with_metadata) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    for (int which = 0; which < 2; ++which) {
        HnswIndex h(4, Metric::L2);
        FlatIndex f(4, Metric::L2);
        for (std::uint64_t i = 0; i < 50; ++i) {
            h.add(i, std::vector<float>(4, float(i)), Metadata().set("old", 1));
            f.add(i, std::vector<float>(4, float(i)), Metadata().set("old", 1));
        }
        const Metadata md = Metadata().set("brand_new_field", "brand_new_string").set_tags("t", {"x", "y"});
        const long points = sweep_allocation_failures(
            [&] { which == 0 ? h.add(999, std::vector<float>(4, 0.5f), md) : f.add(999, std::vector<float>(4, 0.5f), md); },
            [&] {
                const MetadataStore& m = which == 0 ? h.metadata() : f.metadata();
                CHECK(!m.find_field("brand_new_field").has_value() && !m.find_string("brand_new_string").has_value());
                CHECK((which == 0 ? h.size() : f.size()) == 50);
            });
        CHECK(points > 0);
        CHECK(*(which == 0 ? h.get_metadata(999) : f.get_metadata(999)) == md);
    }
}

TEST(metadata, md51_oom_set_metadata) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0}, Metadata().set("a", "x"));
    const long points = sweep_allocation_failures([&] { f.set_metadata(1, Metadata().set("a", "new").set("b", 2)); }, [&] {
        CHECK(*f.get_metadata(1) == Metadata().set("a", "x") && f.metadata().field_count() == 1);
    });
    CHECK(points > 0);
    CHECK(f.get_metadata(1)->get("a")->s == "new");
}

TEST(metadata, md52_oom_reused_slot_with_metadata) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(101, 8, 4, 307);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 100; ++i) h.add(i, data[i], meta_for(i));
    h.remove(7);
    const long points = sweep_allocation_failures([&] { h.add(5000, data[100], Metadata().set("fresh", "value")); }, [&] {
        CHECK(!h.contains(5000) && h.deleted_count() == 1 && !h.metadata().find_field("fresh").has_value());
        check_graph(h, 0.99);
    });
    CHECK(points > 0);
    CHECK(h.get_metadata(5000)->get("fresh")->s == "value" && h.deleted_count() == 0);
}

// ===========================================================================
// Filtered search (FL1-FL50)
// ===========================================================================

namespace {

/// A small Flat index holding meta_for(i) for i in [0, n), vector = {i}.
std::unique_ptr<FlatIndex> small_meta_flat(std::size_t n, bool strict = true) {
    auto f = std::make_unique<FlatIndex>(1, Metric::L2, strict ? Schema::strict(search_fields()) : Schema::dynamic(search_fields()));
    for (std::uint64_t i = 0; i < n; ++i) f->add(i, std::vector<float>{float(i)}, meta_for(i));
    return f;
}

/// IDs the filter matches, in increasing order (via an exhaustive Flat search).
std::vector<std::uint64_t> matching_ids(const FlatIndex& f, const Filter& filter) {
    std::vector<std::uint64_t> ids;
    for (const auto& r : f.search(std::vector<float>{0}, kHuge, with_filter(filter))) ids.push_back(r.id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

/// IDs the reference evaluator says match.
std::vector<std::uint64_t> reference_ids(std::size_t n, const Filter& filter) {
    std::vector<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < n; ++i)
        if (ref_eval(filter.root(), meta_for(i))) ids.push_back(i);
    return ids;
}

}  // namespace

TEST(filter, fl01_comparisons_match_reference) {
    auto f = small_meta_flat(300);
    bool ok = true;
    for (int year : {1999, 2000, 2010, 2024, 2025})
        for (const Filter& flt : {Filter::eq("year", year), Filter::ne("year", year), Filter::lt("year", year),
                                  Filter::le("year", year), Filter::gt("year", year), Filter::ge("year", year),
                                  Filter::between("year", year, year + 5), Filter::lt("price", double(year - 1950)),
                                  Filter::eq("category", "tech"), Filter::ne("category", "tech"), Filter::eq("in_stock", true),
                                  Filter::ne("in_stock", true)})
            ok = ok && matching_ids(*f, flt) == reference_ids(300, flt);
    CHECK(ok);
}

TEST(filter, fl02_membership_and_tags_match_reference) {
    auto f = small_meta_flat(300);
    bool ok = true;
    for (const Filter& flt : {Filter::in("category", {"news", "food"}), Filter::in("year", {2001, 2003, 2024}),
                              Filter::in("in_stock", {false}), Filter::has_tag("tags", "a"),
                              Filter::has_any_tag("tags", {"b", "c"}), Filter::has_all_tags("tags", {"a", "b"}),
                              Filter::has_all_tags("tags", {"a", "b", "c"}), Filter::exists("price"),
                              Filter::exists("category")})
        ok = ok && matching_ids(*f, flt) == reference_ids(300, flt);
    CHECK(ok);
}

TEST(filter, fl03_random_nested_filters_match_reference) {
    auto f = small_meta_flat(400);
    std::mt19937 rng(310);
    bool ok = true;
    for (int i = 0; i < 1000; ++i) {
        const Filter flt = random_filter(rng, 4);
        ok = ok && matching_ids(*f, flt) == reference_ids(400, flt);
    }
    CHECK(ok);
}

TEST(filter, fl04_type_mismatch_rejected) {
    auto f = small_meta_flat(10);
    std::vector<float> q{0};
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::eq("year", "2020"))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::lt("category", "m"))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::lt("tags", 3))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::eq("in_stock", 1))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::has_tag("category", "a"))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::in("tags", {"a"}))));
    CHECK_THROWS_AS(std::invalid_argument, f->search(q, 5, with_filter(Filter::between("category", 1, 2))));
}

TEST(filter, fl05_unknown_field_strict_and_dynamic) {
    auto strict = small_meta_flat(20, true);
    auto dynamic = small_meta_flat(20, false);
    std::vector<float> q{0};
    CHECK_THROWS_AS(std::invalid_argument, strict->search(q, 5, with_filter(Filter::eq("colour", "red"))));
    CHECK(dynamic->search(q, 5, with_filter(Filter::eq("colour", "red"))).empty());
    CHECK(dynamic->search(q, 5, with_filter(!Filter::exists("colour"))).size() == 5);
}

TEST(filter, fl06_unknown_string_matches_nothing) {
    auto f = small_meta_flat(50);
    CHECK(matching_ids(*f, Filter::eq("category", "nonexistent")).empty());
    CHECK(matching_ids(*f, Filter::ne("category", "nonexistent")).size() == 50);  // every vector has a category
    CHECK(matching_ids(*f, Filter::has_tag("tags", "nonexistent")).empty());
}

TEST(filter, fl07_missing_field_semantics) {
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("category", "news"));
    f.add(2, std::vector<float>{1}, Metadata().set("category", "tech"));
    f.add(3, std::vector<float>{2});  // no category
    CHECK(matching_ids(f, Filter::ne("category", "news")) == std::vector<std::uint64_t>{2});   // ne skips missing
    CHECK(matching_ids(f, !Filter::eq("category", "news")) == std::vector<std::uint64_t>({2, 3}));  // ! includes it
    CHECK(matching_ids(f, !Filter::exists("category")) == std::vector<std::uint64_t>{3});
}

TEST(filter, fl08_empty_and_inverted_ranges) {
    auto f = small_meta_flat(50);
    CHECK(matching_ids(*f, Filter::in("category", std::vector<std::string>{})).empty());
    CHECK(matching_ids(*f, Filter::between("year", 2010, 2005)).empty());
    CHECK_THROWS_AS(std::invalid_argument, Filter::eq("price", double(kNaN)));
    CHECK_THROWS_AS(std::invalid_argument, Filter::between("price", 0.0, double(kNaN)));
    CHECK(matching_ids(*f, Filter::lt("price", double(kInf))) == reference_ids(50, Filter::exists("price")));
}

TEST(filter, fl09_exact_integer_float_comparison) {
    FlatIndex f(1, Metric::L2);
    const std::int64_t big = (std::int64_t(1) << 53) + 1;  // not representable as a double
    f.add(1, std::vector<float>{0}, Metadata().set("n", big));
    f.add(2, std::vector<float>{1}, Metadata().set("x", 2.5));
    CHECK(matching_ids(f, Filter::gt("n", double(std::int64_t(1) << 53))).size() == 1);  // exact, not rounded
    CHECK(matching_ids(f, Filter::eq("n", double(std::int64_t(1) << 53))).empty());
    CHECK(matching_ids(f, Filter::lt("x", 3)).size() == 1 && matching_ids(f, Filter::gt("x", 2)).size() == 1);
    CHECK(matching_ids(f, Filter::eq("x", 2)).empty());
    CHECK(matching_ids(f, Filter::lt("n", 1e300)).size() == 1);  // beyond the int64 range
}

TEST(filter, fl10_filter_reusable_across_queries_and_indexes) {
    const Filter flt = Filter::eq("category", "news") && Filter::ge("year", 2010);
    auto a = small_meta_flat(100, true);
    auto b = small_meta_flat(200, false);
    CHECK(matching_ids(*a, flt) == reference_ids(100, flt));
    CHECK(matching_ids(*b, flt) == reference_ids(200, flt));
    CHECK(matching_ids(*a, flt) == reference_ids(100, flt));  // reused again
    FlatIndex other(1, Metric::L2, Schema::strict({{"category", FieldType::Int}}));  // same name, other type
    other.add(1, std::vector<float>{0}, Metadata().set("category", 5));
    CHECK_THROWS_AS(std::invalid_argument, other.search(std::vector<float>{0}, 1, with_filter(flt)));
}

TEST(filter, fl20_flat_filtered_exact) {
    auto data = random_vectors(400, 8, 311);
    auto queries = random_vectors(10, 8, 312);
    std::mt19937 rng(313);
    for (Metric m : kMetrics) {
        FlatIndex f(8, m, Schema::strict(search_fields()));
        for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i], meta_for(i));
        for (int t = 0; t < 10; ++t) {
            const Filter flt = random_filter(rng, 2);
            std::vector<bool> excluded(data.size());
            for (std::size_t i = 0; i < data.size(); ++i) excluded[i] = !ref_eval(flt.root(), meta_for(i));
            for (const auto& q : queries)
                CHECK(same_results(tie_sorted(f.search(q, 15, with_filter(flt))), tie_sorted(reference_search(data, excluded, q, m, 15, 0))));
        }
    }
}

TEST(filter, fl21_fewer_matches_than_k) {
    auto f = small_meta_flat(100);
    const auto r = f->search(std::vector<float>{0}, 50, with_filter(Filter::has_all_tags("tags", {"a", "b", "c"})));
    CHECK(r.size() == reference_ids(100, Filter::has_all_tags("tags", {"a", "b", "c"})).size() && r.size() < 50);
}

TEST(filter, fl22_no_matches) {
    auto f = small_meta_flat(100);
    CHECK(f->search(std::vector<float>{0}, 10, with_filter(Filter::gt("year", 3000))).empty());
}

TEST(filter, fl23_match_everything_equals_unfiltered) {
    auto f = small_meta_flat(100);
    CHECK(same_results(f->search(std::vector<float>{3}, 20, with_filter(Filter::exists("category"))),
                       f->search(std::vector<float>{3}, 20)));
    CHECK(same_results(f->search(std::vector<float>{3}, 20, with_filter(Filter::all())), f->search(std::vector<float>{3}, 20)));
}

TEST(filter, fl24_filters_with_removals) {
    auto f = small_meta_flat(200);
    for (std::uint64_t i = 0; i < 200; i += 4) f->remove(i);
    const Filter flt = Filter::eq("category", "news");
    std::vector<std::uint64_t> expected;
    for (std::uint64_t i : reference_ids(200, flt))
        if (i % 4 != 0) expected.push_back(i);
    CHECK(matching_ids(*f, flt) == expected);
}

TEST(filter, fl30_hnsw_never_violates_filter) {
    const MetaFixture& fx = meta_fixture();
    std::mt19937 rng(314);
    bool ok = true;
    for (int t = 0; t < 1000; ++t) {
        const Filter flt = random_filter(rng, 3);
        for (const auto& r : fx.hnsw->search(fx.queries[std::size_t(t) % fx.queries.size()], 10, 64, with_filter(flt, Strategy::ForceGraph)))
            ok = ok && ref_eval(flt.root(), meta_for(r.id));
    }
    CHECK(ok);
}

TEST(filter, fl31_hnsw_recall_by_selectivity) {
    const MetaFixture& fx = meta_fixture();
    for (int pct : {50, 20, 10, 5}) {
        const double r = filtered_recall(*fx.hnsw, *fx.flat, fx.queries, 10, 64,
                                         with_filter(Filter::lt("bucket", pct), Strategy::ForceGraph));
        CHECK(r >= 0.90);
    }
}

TEST(filter, fl32_hnsw_match_everything_equals_unfiltered) {
    const MetaFixture& fx = meta_fixture();
    bool same = true;
    for (std::size_t i = 0; i < 10; ++i)
        same = same && same_results(fx.hnsw->search(fx.queries[i], 10, 64, with_filter(Filter::exists("category"), Strategy::ForceGraph)),
                                    fx.hnsw->search(fx.queries[i], 10, 64));
    CHECK(same);
}

TEST(filter, fl33_hnsw_match_nothing) {
    const MetaFixture& fx = meta_fixture();
    CHECK(fx.hnsw->search(fx.queries[0], 10, 64, with_filter(Filter::gt("year", 3000), Strategy::ForceGraph)).empty());
    CHECK(fx.hnsw->search(fx.queries[0], 10, 64, with_filter(Filter::gt("year", 3000))).empty());
}

TEST(filter, fl34_hnsw_fewer_matches_than_k) {
    const MetaFixture& fx = meta_fixture();
    const Filter flt = Filter::eq("bucket", 7) && Filter::eq("category", "food");  // a handful of vectors
    const auto expected = fx.flat->search(fx.queries[0], 100, with_filter(flt));
    const auto got = fx.hnsw->search(fx.queries[0], 100, 64, with_filter(flt, Strategy::ForceGraph));
    CHECK(got.size() == expected.size() && expected.size() < 100);
}

TEST(filter, fl35_clustered_versus_scattered_matches) {
    auto [data, label] = clustered_labeled(2000, 16, 8, 315);
    auto queries = clustered(30, 16, 8, 316);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        Metadata md = Metadata().set("cluster", std::int64_t(label[i])).set("scatter", std::int64_t(i % 8));
        h.add(i, data[i], md);
        f.add(i, data[i], md);
    }
    CHECK(filtered_recall(h, f, queries, 10, 64, with_filter(Filter::eq("cluster", 3), Strategy::ForceGraph)) >= 0.85);
    CHECK(filtered_recall(h, f, queries, 10, 64, with_filter(Filter::eq("scatter", 3), Strategy::ForceGraph)) >= 0.90);
}

TEST(filter, fl36_filters_with_removal_and_reuse) {
    auto data = clustered(1000, 8, 6, 317);
    auto queries = clustered(20, 8, 6, 318);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    for (std::uint64_t i = 0; i < 1000; i += 3) {
        h.remove(i);
        f.remove(i);
    }
    for (std::uint64_t i = 0; i < 1000; i += 6) {  // reuse slots with new metadata
        h.add(5000 + i, data[i], meta_for(i + 1));
        f.add(5000 + i, data[i], meta_for(i + 1));
    }
    const SearchOptions o = with_filter(Filter::eq("category", "tech") || Filter::has_tag("tags", "c"), Strategy::ForceGraph);
    CHECK(filtered_recall(h, f, queries, 10, 64, o) >= 0.90);
    for (const auto& q : queries)
        for (const auto& r : h.search(q, 10, 64, o)) CHECK(h.contains(r.id) && ref_eval(o.filter->root(), *h.get_metadata(r.id)));
}

TEST(filter, fl37_entry_point_fails_filter) {
    auto data = clustered(500, 8, 4, 319);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], Metadata().set("ok", true));
    const std::uint64_t entry_id = h.storage().ids().external(h.entry_point());
    h.set_metadata(entry_id, Metadata().set("ok", false));
    const auto r = h.search(data[0], 10, 64, with_filter(Filter::eq("ok", true), Strategy::ForceGraph));
    CHECK(r.size() == 10);
    for (const auto& x : r) CHECK(x.id != entry_id);
}

TEST(filter, fl40_predicate_alone) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions o;
    o.predicate = [](std::uint64_t id) { return id % 3 == 0; };
    for (const auto& q : fx.queries) {
        const auto got = fx.flat->search(q, 10, o);
        for (const auto& r : got) CHECK(r.id % 3 == 0);
        CHECK(got.size() == 10);
    }
    o.strategy = Strategy::ForceGraph;
    CHECK(filtered_recall(*fx.hnsw, *fx.flat, fx.queries, 10, 64, o) >= 0.90);
}

TEST(filter, fl41_predicate_and_filter_both_apply) {
    const MetaFixture& fx = meta_fixture();
    std::atomic<std::size_t> calls{0};
    SearchOptions o = with_filter(Filter::eq("category", "news"));
    o.predicate = [&](std::uint64_t id) {
        ++calls;
        return id % 2 == 0;
    };
    const auto got = fx.flat->search(fx.queries[0], 20, o);
    for (const auto& r : got) CHECK(r.id % 2 == 0 && r.id % 5 == 0);  // news = id % 5 == 0
    CHECK(calls.load() < fx.data.size());  // only called for vectors that passed the filter
}

TEST(filter, fl42_throwing_predicate) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions o;
    o.predicate = [](std::uint64_t id) -> bool {
        if (id == 7) throw std::runtime_error("permission service down");
        return true;
    };
    CHECK_THROWS_AS(std::runtime_error, fx.flat->search(fx.queries[0], 10, o));
    o.strategy = Strategy::ForceExact;
    CHECK_THROWS_AS(std::runtime_error, fx.hnsw->search(fx.queries[0], 10, 64, o));
    CHECK(fx.hnsw->search(fx.queries[0], 10).size() == 10);  // still usable
}

TEST(filter, fl43_constant_predicates) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions none, all;
    none.predicate = [](std::uint64_t) { return false; };
    all.predicate = [](std::uint64_t) { return true; };
    CHECK(fx.hnsw->search(fx.queries[0], 10, 64, none).empty() && fx.flat->search(fx.queries[0], 10, none).empty());
    CHECK(same_results(tie_sorted(fx.flat->search(fx.queries[0], 10, all)), tie_sorted(fx.flat->search(fx.queries[0], 10))));
}

TEST(filter, fl44_predicate_called_concurrently) {
    const MetaFixture& fx = meta_fixture();
    std::atomic<std::size_t> calls{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t id) {
        calls.fetch_add(1, std::memory_order_relaxed);
        return id % 4 != 0;
    };
    const auto buffer = flatten(fx.queries);
    const auto batch = fx.hnsw->search_batch(buffer, 10, 64, 8, o);
    bool ok = true;
    for (std::size_t i = 0; i < batch.size(); ++i) ok = ok && same_results(batch[i], fx.hnsw->search(fx.queries[i], 10, 64, o));
    CHECK(ok && calls.load() > 0);
}

TEST(filter, fl45_predicate_sees_live_user_ids) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(1'000'000 + i, std::vector<float>{float(i), 0});
    for (std::uint64_t i = 0; i < 300; i += 2) h.remove(1'000'000 + i);
    bool only_live = true;
    SearchOptions o;
    o.predicate = [&](std::uint64_t id) {
        only_live = only_live && id >= 1'000'000 && h.contains(id);
        return true;
    };
    o.strategy = Strategy::ForceExact;
    h.search(std::vector<float>{5, 0}, 10, 64, o);
    o.strategy = Strategy::ForceGraph;
    h.search(std::vector<float>{5, 0}, 10, 64, o);
    CHECK(only_live);
}

TEST(filter, fl50_oom_filtered_search) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    const MetaFixture& fx = meta_fixture();
    const SearchOptions o = with_filter(Filter::in("category", {"news", "tech"}) && !Filter::has_tag("tags", "c"));
    const Results before = fx.hnsw->search(fx.queries[0], 10, 64, o);
    Results got;
    const long points = sweep_allocation_failures([&] { got = fx.hnsw->search(fx.queries[0], 10, 64, o); },
                                                  [&] { CHECK(same_results(fx.hnsw->search(fx.queries[0], 10, 64, o), before)); });
    CHECK(points > 0 && same_results(got, before));
}

// ===========================================================================
// Query planner (PL1-PL16)
// ===========================================================================

TEST(planner, pl01_sampled_selectivity_is_accurate) {
    const MetaFixture& fx = meta_fixture();
    for (int pct : {50, 20, 5}) {
        SearchStats st;
        SearchOptions o = with_filter(Filter::lt("bucket", pct));
        o.stats = &st;
        fx.hnsw->search(fx.queries[0], 10, 64, o);
        CHECK(!st.exact_count && std::fabs(st.selectivity - pct / 100.0) < 0.08);
    }
}

TEST(planner, pl02_payload_index_gives_exact_count) {
    auto data = clustered(1000, 8, 4, 320);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    h.create_payload_index("category");
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("category", "music"));
    o.stats = &st;
    h.search(data[0], 10, 64, o);
    CHECK(st.exact_count && st.estimated_matches == reference_ids(1000, Filter::eq("category", "music")).size());
}

TEST(planner, pl03_few_matches_use_exact_search) {
    const MetaFixture& fx = meta_fixture();
    SearchStats st;
    SearchOptions o = with_filter(Filter::lt("bucket", 5));  // ~150 matches
    o.stats = &st;
    fx.hnsw->search(fx.queries[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceExact && st.reason == PlanReason::MatchCount);
}

TEST(planner, pl04_small_fraction_uses_exact_search) {
    const MetaFixture& fx = meta_fixture();
    SearchStats st;
    SearchOptions o = with_filter(Filter::lt("bucket", 5));
    o.planner = PlannerParams{10, 0.10, 256, 32.0};
    o.stats = &st;
    fx.hnsw->search(fx.queries[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceExact && st.reason == PlanReason::MatchFraction);
}

TEST(planner, pl05_otherwise_graph_with_wider_beam) {
    const MetaFixture& fx = meta_fixture();
    SearchStats st;
    SearchOptions o = with_filter(Filter::lt("bucket", 25));
    o.planner = PlannerParams{10, 0.01, 256, 32.0};
    o.stats = &st;
    fx.hnsw->search(fx.queries[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceGraph && st.reason == PlanReason::Graph);
    CHECK(st.ef >= 64 * 3 && st.ef <= 64 * 32);  // about 64 / 0.25
}

TEST(planner, pl06_limits_are_inclusive) {
    auto data = clustered(1000, 8, 4, 321);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    h.create_payload_index("category");
    const std::size_t n = reference_ids(1000, Filter::eq("category", "news")).size();
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("category", "news"));
    o.stats = &st;
    o.planner = PlannerParams{n, 0.0, 256, 32.0};
    h.search(data[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceExact && st.reason == PlanReason::MatchCount);
    o.planner = PlannerParams{n - 1, 0.0, 256, 32.0};
    h.search(data[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceGraph);
    o.planner = PlannerParams{0, double(n) / 1000.0, 256, 32.0};
    h.search(data[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceExact && st.reason == PlanReason::MatchFraction);
}

TEST(planner, pl07_per_query_overrides_index_settings) {
    auto data = clustered(1000, 8, 4, 322);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], Metadata().set("b", std::int64_t(i % 10)));
    h.set_planner_params(PlannerParams{0, 0.0, 256, 32.0});  // index: never exact
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("b", 1));
    o.stats = &st;
    h.search(data[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceGraph);
    o.planner = PlannerParams{5000, 0.5, 256, 32.0};  // query: always exact here
    h.search(data[0], 10, 64, o);
    CHECK(st.strategy == Strategy::ForceExact);
}

TEST(planner, pl08_forced_strategies) {
    const MetaFixture& fx = meta_fixture();
    for (Strategy s : {Strategy::ForceExact, Strategy::ForceGraph}) {
        SearchStats st;
        SearchOptions o = with_filter(Filter::lt("bucket", 30), s);
        o.stats = &st;
        const auto r = fx.hnsw->search(fx.queries[0], 10, 64, o);
        CHECK(st.strategy == s && st.reason == PlanReason::Forced && r.size() == 10);
        check_results(*fx.hnsw, r);
    }
}

TEST(planner, pl09_exact_strategy_equals_flat) {
    auto data = random_vectors(500, 8, 323);
    for (Metric m : kMetrics) {
        HnswIndex h(8, m, HnswParams{}, Schema::strict(search_fields()));
        FlatIndex f(8, m, Schema::strict(search_fields()));
        for (std::size_t i = 0; i < data.size(); ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        const SearchOptions o = with_filter(Filter::ge("year", 2010), Strategy::ForceExact);
        for (std::size_t i = 0; i < 10; ++i)
            CHECK(same_results(tie_sorted(h.search(data[i], 10, 64, o)), tie_sorted(f.search(data[i], 10, o))));
    }
}

TEST(planner, pl10_tiny_index_counts_every_vector) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i, std::vector<float>{float(i), 0}, Metadata().set("b", std::int64_t(i % 4)));
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("b", 1));
    o.stats = &st;
    h.search(std::vector<float>{0, 0}, 5, 64, o);
    CHECK(st.exact_count && st.estimated_matches == 25);
}

TEST(planner, pl11_empty_or_all_removed) {
    HnswIndex h(2, Metric::L2);
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("b", 1));
    o.stats = &st;
    CHECK(h.search(std::vector<float>{0, 0}, 5, 64, o).empty());
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, std::vector<float>{float(i), 0}, Metadata().set("b", 1));
    for (std::uint64_t i = 0; i < 20; ++i) h.remove(i);
    CHECK(h.search(std::vector<float>{0, 0}, 5, 64, o).empty());
}

TEST(planner, pl12_predicate_selectivity_sampled) {
    const MetaFixture& fx = meta_fixture();
    SearchStats st;
    SearchOptions o;
    o.predicate = [](std::uint64_t id) { return id % 4 == 0; };
    o.stats = &st;
    fx.hnsw->search(fx.queries[0], 10, 64, o);
    CHECK(!st.exact_count && std::fabs(st.selectivity - 0.25) < 0.08);
}

TEST(planner, pl13_planner_is_deterministic) {
    const MetaFixture& fx = meta_fixture();
    SearchStats a, b;
    SearchOptions o = with_filter(Filter::lt("bucket", 40));
    o.stats = &a;
    const auto ra = fx.hnsw->search(fx.queries[1], 10, 64, o);
    o.stats = &b;
    const auto rb = fx.hnsw->search(fx.queries[1], 10, 64, o);
    CHECK(same_results(ra, rb) && a.selectivity == b.selectivity && a.strategy == b.strategy && a.ef == b.ef);
}

TEST(planner, pl14_stats_are_consistent) {
    const MetaFixture& fx = meta_fixture();
    for (Strategy s : {Strategy::ForceExact, Strategy::ForceGraph}) {
        SearchStats st;
        SearchOptions o = with_filter(Filter::lt("bucket", 30), s);
        o.stats = &st;
        const auto r = fx.hnsw->search(fx.queries[2], 10, 64, o);
        CHECK(st.distance_computations >= r.size() && st.nodes_visited > 0);
    }
    SearchStats flat;
    SearchOptions o = with_filter(Filter::lt("bucket", 30));
    o.stats = &flat;
    fx.flat->search(fx.queries[2], 10, o);
    CHECK(flat.reason == PlanReason::FlatIndex && flat.estimated_matches == 900 && flat.distance_computations == 900);
}

TEST(planner, pl15_invalid_settings_rejected) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    CHECK_THROWS_AS(std::invalid_argument, h.set_planner_params(PlannerParams{10, 1.5, 256, 32.0}));
    CHECK_THROWS_AS(std::invalid_argument, h.set_planner_params(PlannerParams{10, double(kNaN), 256, 32.0}));
    CHECK_THROWS_AS(std::invalid_argument, h.set_planner_params(PlannerParams{10, 0.1, 0, 32.0}));
    CHECK_THROWS_AS(std::invalid_argument, h.set_planner_params(PlannerParams{10, 0.1, 256, 0.5}));
    SearchOptions o;
    o.planner = PlannerParams{10, -0.1, 256, 32.0};
    CHECK_THROWS_AS(std::invalid_argument, h.search(std::vector<float>{0, 0}, 1, 64, o));
}

TEST(planner, pl16_auto_keeps_recall_across_selectivities) {
    const MetaFixture& fx = meta_fixture();
    for (int pct : {1, 2, 5, 10, 25, 50, 100}) {
        const SearchOptions o = with_filter(Filter::lt("bucket", pct));
        CHECK(filtered_recall(*fx.hnsw, *fx.flat, fx.queries, 10, 64, o) >= 0.90);
        for (const auto& q : fx.queries)
            for (const auto& r : fx.hnsw->search(q, 10, 64, o)) CHECK(int(r.id % 100) < pct);
    }
}

// ===========================================================================
// Payload index (PI1-PI6)
// ===========================================================================

TEST(payload, pi01_counts_match_data_after_churn) {
    HnswIndex h(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    h.create_payload_index("category");
    h.create_payload_index("tags");
    h.create_payload_index("in_stock");
    std::mt19937 rng(324);
    std::vector<std::uint64_t> live;
    std::uint64_t next = 0;
    for (int step = 0; step < 2000; ++step) {
        const unsigned op = static_cast<unsigned>(rng() % 4);
        if (op < 2 || live.empty()) {
            h.add(next, std::vector<float>{float(next), 0}, meta_for(next));
            live.push_back(next++);
        } else if (op == 2) {
            const std::size_t pos = rng() % live.size();
            h.remove(live[pos]);
            live[pos] = live.back();
            live.pop_back();
        } else {
            h.set_metadata(live[rng() % live.size()], meta_for(rng() % 1000));
        }
    }
    const MetadataStore& m = h.metadata();
    bool ok = true;
    for (const char* cat : kCategories) {
        std::size_t expected = 0;
        for (auto id : live) expected += h.get_metadata(id)->get("category")->s == cat;
        ok = ok && m.value_count(*m.find_field("category"), *m.find_string(cat)) == expected;
    }
    for (const char* tag : {"a", "b", "c"}) {
        std::size_t expected = 0;
        for (auto id : live) {
            auto t = h.get_metadata(id)->get("tags");
            expected += t && std::find(t->tags.begin(), t->tags.end(), tag) != t->tags.end();
        }
        ok = ok && m.value_count(*m.find_field("tags"), *m.find_string(tag)) == expected;
    }
    std::size_t in_stock = 0;
    for (auto id : live) {
        auto v = h.get_metadata(id)->get("in_stock");
        in_stock += v && v->b;
    }
    ok = ok && m.value_count(*m.find_field("in_stock"), 1) == in_stock;
    CHECK(ok);
}

TEST(payload, pi02_identical_results_with_and_without_index) {
    auto data = clustered(1500, 8, 6, 325);
    HnswIndex a(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    HnswIndex b(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    b.create_payload_index("category");
    for (std::size_t i = 0; i < data.size(); ++i) {
        a.add(i, data[i], meta_for(i));
        b.add(i, data[i], meta_for(i));
    }
    const SearchOptions o = with_filter(Filter::eq("category", "food"), Strategy::ForceExact);
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) same = same && same_results(a.search(data[i], 10, 64, o), b.search(data[i], 10, 64, o));
    CHECK(same);
}

TEST(payload, pi03_backfill_equals_incremental) {
    HnswIndex a(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    HnswIndex b(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    a.create_payload_index("category");
    for (std::uint64_t i = 0; i < 300; ++i) {
        a.add(i, std::vector<float>{float(i), 0}, meta_for(i));
        b.add(i, std::vector<float>{float(i), 0}, meta_for(i));
    }
    b.create_payload_index("category");  // built over existing data
    bool same = true;
    for (const char* cat : kCategories)
        same = same && a.metadata().value_count(0, *a.metadata().find_string(cat)) ==
                           b.metadata().value_count(0, *b.metadata().find_string(cat));
    CHECK(same);
}

TEST(payload, pi04_unsupported_fields_rejected) {
    FlatIndex f(2, Metric::L2, Schema::strict(search_fields()));
    CHECK_THROWS_AS(std::invalid_argument, f.create_payload_index("year"));
    CHECK_THROWS_AS(std::invalid_argument, f.create_payload_index("price"));
    CHECK_THROWS_AS(std::invalid_argument, f.create_payload_index("nope"));
    CHECK_NOTHROW(f.create_payload_index("category"));
    CHECK_NOTHROW(f.create_payload_index("category"));  // twice is harmless
}

TEST(payload, pi05_oom_index_updates) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    FlatIndex f(2, Metric::L2, Schema::strict(search_fields()));
    for (std::uint64_t i = 0; i < 50; ++i) f.add(i, std::vector<float>{float(i), 0}, meta_for(i));
    CHECK(sweep_allocation_failures([&] { f.create_payload_index("category"); },
                                    [&] { CHECK(!f.metadata().is_indexed(0)); }) > 0);
    const std::size_t field = *f.metadata().find_field("tags");
    f.create_payload_index("tags");
    CHECK(sweep_allocation_failures([&] { f.add(999, std::vector<float>{0, 0}, Metadata().set_tags("tags", {"zz_new"})); },
                                    [&] { CHECK(!f.metadata().find_string("zz_new").has_value()); }) > 0);
    CHECK(f.metadata().value_count(field, *f.metadata().find_string("zz_new")) == 1);
}

TEST(payload, pi06_compact_keeps_index) {
    HnswIndex h(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    h.create_payload_index("category");
    for (std::uint64_t i = 0; i < 200; ++i) h.add(i, std::vector<float>{float(i), 0}, meta_for(i));
    for (std::uint64_t i = 0; i < 200; i += 2) h.remove(i);
    h.compact();
    const MetadataStore& m = h.metadata();
    REQUIRE(m.is_indexed(*m.find_field("category")));
    std::size_t news = 0;
    for (std::uint64_t i = 1; i < 200; i += 2) news += i % 5 == 0;
    CHECK(m.value_count(*m.find_field("category"), *m.find_string("news")) == news);
}

// ===========================================================================
// Batch search (BT1-BT15)
// ===========================================================================

TEST(batch, bt01_hnsw_batch_equals_single_searches) {
    for (Metric m : kMetrics) {
        const Fixture& fx = fixture(m);
        const auto batch = fx.hnsw->search_batch(flatten(fx.queries), 10, 64, 4);
        bool same = batch.size() == fx.queries.size();
        for (std::size_t i = 0; same && i < batch.size(); ++i) same = same_results(batch[i], fx.hnsw->search(fx.queries[i], 10, 64));
        CHECK(same);
    }
}

TEST(batch, bt02_flat_tiled_batch_equals_single_searches) {
    for (Metric m : kMetrics) {
        const Fixture& fx = fixture(m);
        const auto batch = fx.flat->search_batch(flatten(fx.queries), 10, 4);
        bool same = batch.size() == fx.queries.size();
        for (std::size_t i = 0; same && i < batch.size(); ++i) same = same_results(batch[i], fx.flat->search(fx.queries[i], 10));
        CHECK(same);
    }
}

TEST(batch, bt03_thread_count_does_not_change_results) {
    const Fixture& fx = fixture(Metric::L2);
    const auto buffer = flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 3));
    const auto h1 = fx.hnsw->search_batch(buffer, 10, 64, 1);
    const auto f1 = fx.flat->search_batch(buffer, 10, 1);
    for (std::size_t threads : std::initializer_list<std::size_t>{2, 4, 8, 64}) {  // 64: more threads than queries
        const auto h = fx.hnsw->search_batch(buffer, 10, 64, threads);
        const auto f = fx.flat->search_batch(buffer, 10, threads);
        for (std::size_t i = 0; i < 3; ++i) CHECK(same_results(h[i], h1[i]) && same_results(f[i], f1[i]));
    }
}

TEST(batch, bt04_zero_and_one_query) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK(fx.hnsw->search_batch(std::vector<float>{}, 10).empty());
    CHECK(fx.flat->search_batch(std::vector<float>{}, 10).empty());
    const auto one = fx.hnsw->search_batch(fx.queries[0], 10);
    CHECK(one.size() == 1 && same_results(one[0], fx.hnsw->search(fx.queries[0], 10, 64)));
}

TEST(batch, bt05_buffer_not_multiple_of_dimension) {
    const Fixture& fx = fixture(Metric::L2);
    std::vector<float> bad(32 * 2 + 5, 0.1f);
    CHECK_THROWS_AS(std::invalid_argument, fx.hnsw->search_batch(bad, 10));
    CHECK_THROWS_AS(std::invalid_argument, fx.flat->search_batch(bad, 10));
}

TEST(batch, bt06_one_invalid_query_rejects_whole_batch) {
    const Fixture& fx = fixture(Metric::L2);
    auto buffer = flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 5));
    buffer[3 * 32 + 7] = kNaN;
    std::atomic<int> calls{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t) { ++calls; return true; };
    CHECK_THROWS_AS(std::invalid_argument, fx.hnsw->search_batch(buffer, 10, 64, 4, o));
    CHECK_THROWS_AS(std::invalid_argument, fx.flat->search_batch(buffer, 10, 4, o));
    CHECK(calls.load() == 0);  // rejected before any work
}

TEST(batch, bt07_shared_filter_and_predicate) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions o = with_filter(Filter::lt("bucket", 30));
    o.predicate = [](std::uint64_t id) { return id % 2 == 1; };
    const auto buffer = flatten(fx.queries);
    const auto hb = fx.hnsw->search_batch(buffer, 10, 64, 4, o);
    const auto fb = fx.flat->search_batch(buffer, 10, 4, o);
    bool same = true;
    for (std::size_t i = 0; i < fx.queries.size(); ++i)
        same = same && same_results(hb[i], fx.hnsw->search(fx.queries[i], 10, 64, o)) &&
               same_results(fb[i], fx.flat->search(fx.queries[i], 10, o));
    CHECK(same);
}

TEST(batch, bt08_k_and_ef_extremes) {
    const Fixture& fx = fixture(Metric::L2);
    const auto buffer = flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 3));
    for (const auto& r : fx.hnsw->search_batch(buffer, 0, 64, 2)) CHECK(r.empty());
    for (const auto& r : fx.flat->search_batch(buffer, 0, 2)) CHECK(r.empty());
    for (const auto& r : fx.flat->search_batch(buffer, kHuge, 2)) CHECK(r.size() == 3000);
    for (const auto& r : fx.hnsw->search_batch(buffer, 5, 0, 2)) CHECK(r.size() == 5);
    const auto huge = fx.hnsw->search_batch(buffer, 20, kHuge, 2);
    for (std::size_t i = 0; i < 3; ++i) CHECK(same_results(huge[i], fx.hnsw->search(fx.queries[i], 20, kHuge)));
}

TEST(batch, bt09_empty_or_all_removed) {
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    const std::vector<float> buffer(8, 0.5f);
    for (const auto& r : h.search_batch(buffer, 5)) CHECK(r.empty());
    for (const auto& r : f.search_batch(buffer, 5)) CHECK(r.empty());
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, std::vector<float>(4, float(i)));
        f.add(i, std::vector<float>(4, float(i)));
    }
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.remove(i);
        f.remove(i);
    }
    for (const auto& r : h.search_batch(buffer, 5, 64, 2)) CHECK(r.empty());
    for (const auto& r : f.search_batch(buffer, 5, 2)) CHECK(r.empty());
}

TEST(batch, bt10_tile_boundaries) {
    // Flat tiles 256 vectors by 16 queries; test counts around both sizes.
    for (std::size_t n : std::initializer_list<std::size_t>{1, 255, 256, 257, 600}) {
        auto data = random_vectors(n, 8, unsigned(330 + n));
        FlatIndex f(8, Metric::L2);
        for (std::size_t i = 0; i < n; ++i) f.add(i, data[i]);
        for (std::size_t nq : std::initializer_list<std::size_t>{1, 15, 16, 17, 40}) {
            const auto queries = random_vectors(nq, 8, unsigned(340 + nq));
            const auto batch = f.search_batch(flatten(queries), 7, 3);
            bool same = batch.size() == nq;
            for (std::size_t i = 0; same && i < nq; ++i) same = same_results(batch[i], f.search(queries[i], 7));
            CHECK(same);
        }
    }
}

TEST(batch, bt11_large_batch_and_pool_reuse) {
    const Fixture& fx = fixture(Metric::L2);
    const auto queries = clustered(10000, 32, 20, 350);
    const auto buffer = flatten(queries);
    const auto first = fx.hnsw->search_batch(buffer, 5, 32, 4);
    const auto second = fx.hnsw->search_batch(buffer, 5, 32, 4);  // reuses the same pool
    bool same = first.size() == 10000;
    for (std::size_t i = 0; same && i < first.size(); i += 97) same = same_results(first[i], second[i]) && same_results(first[i], fx.hnsw->search(queries[i], 5, 32));
    CHECK(same);
    SharedPool pool;
    auto a = pool.get(3), b = pool.get(3);
    CHECK(a == b && a->size() == 3);  // reused, not recreated
    CHECK(pool.get(2)->size() == 2);  // recreated for a different size
}

TEST(batch, bt12_failing_task_keeps_pool_usable) {
    ThreadPool pool(4);
    CHECK_THROWS_AS(std::runtime_error, pool.parallel_for(100, [](std::size_t i) {
        if (i == 42) throw std::runtime_error("task failed");
    }));
    std::atomic<std::size_t> sum{0};
    pool.parallel_for(1000, [&](std::size_t i) { sum += i; });
    CHECK(sum.load() == 1000 * 999 / 2);
}

TEST(batch, bt13_pool_stress_and_shutdown) {
    for (int round = 0; round < 50; ++round) {
        ThreadPool pool(1 + std::size_t(round % 6));
        std::atomic<std::size_t> count{0};
        for (int cycle = 0; cycle < 20; ++cycle) pool.parallel_for(37, [&](std::size_t) { ++count; });
        CHECK(count.load() == 20 * 37);
    }  // each pool is destroyed here, joining its workers
    ThreadPool shared(4);
    std::atomic<std::size_t> total{0};
    std::vector<std::thread> callers;
    for (int t = 0; t < 4; ++t)  // several threads use one pool at the same time
        callers.emplace_back([&] { shared.parallel_for(500, [&](std::size_t) { ++total; }); });
    for (auto& c : callers) c.join();
    CHECK(total.load() == 2000);
}

TEST(batch, bt14_batches_alongside_other_searches) {
    const Fixture& fx = fixture(Metric::Cosine);
    std::vector<Results> expected;
    for (const auto& q : fx.queries) expected.push_back(fx.hnsw->search(q, 10, 64));
    std::atomic<int> mismatches{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 3; ++t)
        threads.emplace_back([&] {
            const auto batch = fx.hnsw->search_batch(flatten(fx.queries), 10, 64, 3);
            for (std::size_t i = 0; i < batch.size(); ++i)
                if (!same_results(batch[i], expected[i])) ++mismatches;
        });
    for (int t = 0; t < 3; ++t)
        threads.emplace_back([&] {
            for (std::size_t i = 0; i < fx.queries.size(); ++i)
                if (!same_results(fx.hnsw->search(fx.queries[i], 10, 64), expected[i])) ++mismatches;
        });
    for (auto& t : threads) t.join();
    CHECK(mismatches.load() == 0);
}

TEST(batch, bt15_oom_batch) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 8, 4, 351);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const auto buffer = flatten(Vectors(data.begin(), data.begin() + 4));
    const auto hb = h.search_batch(buffer, 5, 64, 1), fb = f.search_batch(buffer, 5, 1);
    CHECK(sweep_allocation_failures([&] { h.search_batch(buffer, 5, 64, 1); },
                                    [&] { CHECK(same_results(h.search_batch(buffer, 5, 64, 1)[2], hb[2])); }) > 0);
    CHECK(sweep_allocation_failures([&] { f.search_batch(buffer, 5, 1); },
                                    [&] { CHECK(same_results(f.search_batch(buffer, 5, 1)[2], fb[2])); }) > 0);
}

// ===========================================================================
// Range search (RG1-RG15)
// ===========================================================================

namespace {

/// Every eligible vector within `radius`, by brute force, sorted by distance then ID.
Results range_reference(const FlatIndex& f, const std::vector<float>& q, float radius, const SearchOptions& o = {}) {
    Results all;
    for (const auto& r : f.search(q, kHuge, o))
        if (r.distance <= radius) all.push_back(r);
    return tie_sorted(all);
}

double range_recall(const HnswIndex& h, const FlatIndex& f, const Vectors& queries, float radius,
                    const SearchOptions& o = {}) {
    double total = 0.0;
    std::size_t counted = 0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : range_reference(f, q, radius, o)) truth.insert(r.id);
        if (truth.empty()) continue;
        std::size_t hits = 0;
        for (const auto& r : h.search_range(q, radius, kNoLimit, o)) hits += truth.count(r.id);
        total += double(hits) / double(truth.size());
        ++counted;
    }
    return counted ? total / double(counted) : 1.0;
}

/// Checks the range invariants: sorted, unique, live, all inside the radius.
template <class Index>
void check_range(const Index& index, const Results& r, float radius) {
    check_results(index, r);
    bool inside = true;
    for (const auto& x : r) inside = inside && x.distance <= radius;
    CHECK(inside);
}

}  // namespace

TEST(range, rg01_flat_equals_reference) {
    auto data = random_vectors(500, 8, 360);
    for (Metric m : kMetrics) {
        FlatIndex f(8, m);
        for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
        const float radius = m == Metric::L2 ? 2.0f : (m == Metric::Cosine ? 0.4f : -0.5f);
        for (std::size_t i = 0; i < 10; ++i) {
            const auto got = f.search_range(data[i], radius);
            CHECK(same_results(tie_sorted(got), range_reference(f, data[i], radius)));
            check_range(f, got, radius);
        }
    }
}

TEST(range, rg02_l2_uses_squared_radius) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{3, 0});  // distance 3, squared 9
    f.add(2, std::vector<float>{5, 0});  // distance 5, squared 25
    CHECK(f.search_range(std::vector<float>{0, 0}, 4.0f).empty());  // 4 is a squared radius: distance 2
    const auto r = f.search_range(std::vector<float>{0, 0}, l2_radius(4.0f));
    CHECK(r.size() == 1 && r[0].id == 1 && r[0].distance == 9.0f);
    CHECK(l2_radius(3.0f) == 9.0f);
}

TEST(range, rg03_radius_zero) {
    FlatIndex f(2, Metric::L2);
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) {
        f.add(i, std::vector<float>{float(i % 10), 0});
        h.add(i, std::vector<float>{float(i % 10), 0});
    }
    const auto rf = f.search_range(std::vector<float>{3, 0}, 0.0f);
    const auto rh = h.search_range(std::vector<float>{3, 0}, 0.0f);
    CHECK(rf.size() == 5 && rh.size() == 5);  // the five exact copies
    for (const auto& x : rf) CHECK(x.id % 10 == 3 && x.distance == 0.0f);
}

TEST(range, rg04_negative_radius) {
    FlatIndex l2(2, Metric::L2), ip(2, Metric::InnerProduct);
    for (std::uint64_t i = 1; i <= 5; ++i) {
        l2.add(i, std::vector<float>{float(i), 0});
        ip.add(i, std::vector<float>{float(i), 0});
    }
    CHECK(l2.search_range(std::vector<float>{0, 0}, -1.0f).empty());
    const auto r = ip.search_range(std::vector<float>{1, 0}, -3.0f);  // dot product >= 3
    CHECK(r.size() == 3 && r[0].id == 5 && r[2].id == 3);
}

TEST(range, rg05_everything_without_cap) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK(fx.flat->search_range(fx.queries[0], 1e30f).size() == 3000);
    CHECK(fx.hnsw->search_range(fx.queries[0], 1e30f).size() == 3000);
}

TEST(range, rg06_max_results_keeps_the_closest) {
    const Fixture& fx = fixture(Metric::L2);
    const auto all = fx.flat->search_range(fx.queries[0], 1e30f);
    const auto capped = fx.flat->search_range(fx.queries[0], 1e30f, 25);
    CHECK(same_results(capped, Results(all.begin(), all.begin() + 25)));
    const auto hcapped = fx.hnsw->search_range(fx.queries[0], 1e30f, 25);
    CHECK(hcapped.size() == 25);
    check_range(*fx.hnsw, hcapped, 1e30f);
}

TEST(range, rg07_max_results_zero) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK(fx.flat->search_range(fx.queries[0], 1e30f, 0).empty());
    CHECK(fx.hnsw->search_range(fx.queries[0], 1e30f, 0).empty());
}

TEST(range, rg08_nan_and_infinite_radius) {
    const Fixture& fx = fixture(Metric::L2);
    CHECK_THROWS_AS(std::invalid_argument, fx.flat->search_range(fx.queries[0], kNaN));
    CHECK_THROWS_AS(std::invalid_argument, fx.hnsw->search_range(fx.queries[0], kNaN));
    CHECK(fx.flat->search_range(fx.queries[0], kInf).size() == 3000);
    CHECK(fx.hnsw->search_range(fx.queries[0], kInf).size() == 3000);
}

TEST(range, rg09_hnsw_recall_by_radius) {
    const Fixture& fx = fixture(Metric::L2);
    for (float radius : {1.0f, 2.0f, 4.0f}) {
        CHECK(range_recall(*fx.hnsw, *fx.flat, fx.queries, radius) >= 0.90);
        for (std::size_t i = 0; i < 10; ++i) check_range(*fx.hnsw, fx.hnsw->search_range(fx.queries[i], radius), radius);
    }
}

TEST(range, rg10_crossing_outside_vectors_helps) {
    const Fixture& fx = fixture(Metric::L2);
    SearchOptions none, two;
    none.range_expand_outside = 0;
    two.range_expand_outside = 2;
    const double r0 = range_recall(*fx.hnsw, *fx.flat, fx.queries, 3.0f, none);
    const double r2 = range_recall(*fx.hnsw, *fx.flat, fx.queries, 3.0f, two);
    CHECK(r2 >= r0 && r2 >= 0.95);
}

TEST(range, rg11_range_with_filter_and_predicate) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions o = with_filter(Filter::lt("bucket", 50));
    o.predicate = [](std::uint64_t id) { return id % 2 == 0; };
    for (std::size_t i = 0; i < 5; ++i) {
        const auto got = fx.flat->search_range(fx.queries[i], 3.0f, kNoLimit, o);
        CHECK(same_results(tie_sorted(got), range_reference(*fx.flat, fx.queries[i], 3.0f, o)));
        for (const auto& r : fx.hnsw->search_range(fx.queries[i], 3.0f, kNoLimit, o)) CHECK(r.id % 100 < 50 && r.id % 2 == 0);
    }
    CHECK(range_recall(*fx.hnsw, *fx.flat, fx.queries, 3.0f, o) >= 0.90);
}

TEST(range, rg12_never_returns_removed) {
    auto data = clustered(600, 8, 4, 361);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (std::uint64_t i = 0; i < 600; i += 2) {
        h.remove(i);
        f.remove(i);
    }
    for (std::uint64_t i = 0; i < 600; i += 10) {  // reuse some slots
        h.add(9000 + i, data[i]);
        f.add(9000 + i, data[i]);
    }
    for (std::size_t i = 0; i < 20; ++i) {
        check_range(h, h.search_range(data[i], 1.0f), 1.0f);
        check_range(f, f.search_range(data[i], 1.0f), 1.0f);
    }
}

TEST(range, rg13_forced_exact_matches_flat) {
    const MetaFixture& fx = meta_fixture();
    SearchOptions o;
    o.strategy = Strategy::ForceExact;
    for (std::size_t i = 0; i < 10; ++i)
        CHECK(same_results(tie_sorted(fx.hnsw->search_range(fx.queries[i], 2.5f, kNoLimit, o)),
                           range_reference(*fx.flat, fx.queries[i], 2.5f)));
}

TEST(range, rg14_empty_or_all_removed) {
    HnswIndex h(2, Metric::L2);
    FlatIndex f(2, Metric::L2);
    CHECK(h.search_range(std::vector<float>{0, 0}, 10.0f).empty() && f.search_range(std::vector<float>{0, 0}, 10.0f).empty());
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, std::vector<float>{float(i), 0});
        f.add(i, std::vector<float>{float(i), 0});
    }
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.remove(i);
        f.remove(i);
    }
    CHECK(h.search_range(std::vector<float>{0, 0}, 1e30f).empty() && f.search_range(std::vector<float>{0, 0}, 1e30f).empty());
}

TEST(range, rg15_oom_range_search) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 8, 4, 362);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Results before = h.search_range(data[0], 1.0f);
    CHECK(sweep_allocation_failures([&] { h.search_range(data[0], 1.0f); },
                                    [&] { CHECK(same_results(h.search_range(data[0], 1.0f), before)); }) > 0);
}

// ===========================================================================
// Search features together (IX1-IX6)
// ===========================================================================

TEST(search_e2e, ix01_lifecycle_with_filters) {
    auto pool = clustered(2500, 12, 8, 370);
    auto queries = clustered(20, 12, 8, 371);
    HnswIndex h(12, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    FlatIndex f(12, Metric::L2, Schema::strict(search_fields()));
    std::mt19937 rng(372);
    std::vector<std::uint64_t> live;
    std::size_t next = 0;
    const SearchOptions o = with_filter(Filter::eq("category", "tech") || Filter::ge("year", 2020));
    double worst = 1.0;
    for (int step = 0; step < 4000 && next < pool.size(); ++step) {
        const unsigned op = static_cast<unsigned>(rng() % 6);
        if (op < 3 || live.size() < 50) {
            h.add(next, pool[next], meta_for(next));
            f.add(next, pool[next], meta_for(next));
            live.push_back(next++);
        } else if (op == 3) {
            const std::size_t pos = rng() % live.size();
            h.remove(live[pos]);
            f.remove(live[pos]);
            live[pos] = live.back();
            live.pop_back();
        } else if (op == 4) {
            const std::uint64_t id = live[rng() % live.size()];
            const Metadata md = meta_for(rng() % 5000);
            h.set_metadata(id, md);
            f.set_metadata(id, md);
        } else if (step % 50 == 0) {
            worst = std::min(worst, filtered_recall(h, f, queries, 10, 64, o));
            for (const auto& q : queries)
                for (const auto& r : h.search(q, 10, 64, o)) CHECK(ref_eval(o.filter->root(), *h.get_metadata(r.id)));
        }
        if (step == 2000) h.compact();
    }
    CHECK(worst >= 0.85);
    check_graph(h, 0.99);
}

TEST(search_e2e, ix02_batches_with_filters_during_churn) {
    auto pool = clustered(1500, 8, 6, 373);
    auto queries = clustered(16, 8, 6, 374);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 1000; ++i) h.add(i, pool[i], meta_for(i));
    const SearchOptions o = with_filter(Filter::has_tag("tags", "a"));
    bool same = true;
    for (std::size_t round = 0; round < 5; ++round) {
        for (std::size_t i = 0; i < 100; ++i) h.remove(round * 100 + i);
        for (std::size_t i = 0; i < 100; ++i) h.add(1000 + round * 100 + i, pool[1000 + round * 100 + i], meta_for(i));
        const auto batch = h.search_batch(flatten(queries), 10, 64, 4, o);
        for (std::size_t q = 0; q < queries.size(); ++q) same = same && same_results(batch[q], h.search(queries[q], 10, 64, o));
    }
    CHECK(same);
}

TEST(search_e2e, ix03_filtered_searches_from_many_threads) {
    const MetaFixture& fx = meta_fixture();
    const SearchOptions o = with_filter(Filter::lt("bucket", 30) && Filter::ne("category", "food"));
    std::vector<Results> expected;
    for (const auto& q : fx.queries) expected.push_back(fx.hnsw->search(q, 10, 64, o));
    CHECK(parallel_mismatches(fx.queries, expected, 8, 2, [&](const std::vector<float>& q) { return fx.hnsw->search(q, 10, 64, o); }) == 0);
}

TEST(search_e2e, ix04_every_metric_with_every_feature) {
    auto data = clustered(800, 8, 4, 375);
    for (Metric m : kMetrics) {
        HnswIndex h(8, m, HnswParams{}, Schema::strict(search_fields()));
        FlatIndex f(8, m, Schema::strict(search_fields()));
        for (std::size_t i = 0; i < data.size(); ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        const SearchOptions o = with_filter(Filter::ge("year", 2005));
        const Vectors queries(data.begin(), data.begin() + 10);
        CHECK(filtered_recall(h, f, queries, 10, 64, o) >= (m == Metric::InnerProduct ? 0.80 : 0.90));
        const auto batch = h.search_batch(flatten(queries), 10, 64, 2, o);
        for (std::size_t i = 0; i < queries.size(); ++i) CHECK(same_results(batch[i], h.search(queries[i], 10, 64, o)));
        const float radius = m == Metric::L2 ? 0.5f : (m == Metric::Cosine ? 0.05f : -1.0f);
        for (std::size_t i = 0; i < 3; ++i) check_range(h, h.search_range(queries[i], radius, kNoLimit, o), radius);
    }
}

TEST(search_e2e, ix05_reproducible) {
    auto data = clustered(700, 8, 4, 376);
    HnswIndex a(8, Metric::L2), b(8, Metric::L2);
    for (HnswIndex* h : {&a, &b}) {
        for (std::size_t i = 0; i < data.size(); ++i) h->add(i, data[i], meta_for(i));
        for (std::uint64_t i = 0; i < 700; i += 5) h->remove(i);
    }
    const SearchOptions o = with_filter(Filter::has_any_tag("tags", {"b", "c"}));
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) {
        same = same && same_results(a.search(data[i], 10, 64, o), b.search(data[i], 10, 64, o));
        same = same && same_results(a.search_range(data[i], 0.5f, kNoLimit, o), b.search_range(data[i], 0.5f, kNoLimit, o));
    }
    const auto ba = a.search_batch(flatten(data), 5, 64, 3, o), bb = b.search_batch(flatten(data), 5, 64, 1, o);
    for (std::size_t i = 0; i < ba.size(); ++i) same = same && same_results(ba[i], bb[i]);
    CHECK(same);
}

TEST(search_e2e, ix06_large_scale_random_filters) {
    auto data = clustered(20000, 16, 25, 377);
    auto queries = clustered(30, 16, 25, 378);
    HnswIndex h(16, Metric::L2, HnswParams{16, 100, 9}, Schema::strict(search_fields()));
    FlatIndex f(16, Metric::L2, Schema::strict(search_fields()));
    h.create_payload_index("category");
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    std::mt19937 rng(379);
    double total = 0.0;
    int tested = 0;
    for (int t = 0; t < 40; ++t) {
        const SearchOptions o = with_filter(random_filter(rng, 2));
        for (const auto& r : h.search(queries[std::size_t(t) % queries.size()], 10, 64, o))
            CHECK(ref_eval(o.filter->root(), meta_for(r.id)));
        total += filtered_recall(h, f, Vectors(queries.begin(), queries.begin() + 5), 10, 64, o);
        ++tested;
    }
    CHECK(total / tested >= 0.90);
}

// ===========================================================================
// Added by the second coverage audit (beyond the test plan)
// ===========================================================================

TEST(filter, fl60_compile_errors_for_every_type) {
    auto f = small_meta_flat(10);
    const std::vector<float> q{0};
    for (const Filter& bad : {Filter::eq("tags", "a"), Filter::in("in_stock", {"yes"}), Filter::in("year", {"x"}),
                              Filter::in("category", {1, 2}), Filter::has_all_tags("category", {"a"}),
                              Filter::between("year", "a", 3), Filter::between("year", 1, "b"), Filter::lt("in_stock", true),
                              Filter::ne("category", 3), Filter::exists("unknown_field"), Filter::in("price", {true})})
        CHECK_THROWS_AS(std::invalid_argument, f->search(q, 1, with_filter(bad)));
}

TEST(filter, fl61_constant_folding) {
    auto f = small_meta_flat(50, false);  // dynamic: unknown fields match nothing
    const Filter unknown = Filter::eq("missing", 1), news = Filter::eq("category", "news");
    CHECK(matching_ids(*f, !Filter::all()).empty());
    CHECK(matching_ids(*f, !!Filter::all()).size() == 50);
    CHECK(matching_ids(*f, !unknown).size() == 50);
    CHECK(matching_ids(*f, unknown && news).empty() && matching_ids(*f, news && unknown).empty());
    CHECK(matching_ids(*f, unknown || news) == reference_ids(50, news));
    CHECK(matching_ids(*f, news || unknown) == reference_ids(50, news));
    CHECK(matching_ids(*f, unknown || unknown).empty());
    CHECK(matching_ids(*f, !(unknown || unknown)).size() == 50);
}

TEST(filter, fl62_number_comparison_edges) {
    FlatIndex f(1, Metric::L2);
    const std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    f.add(1, std::vector<float>{0}, Metadata().set("n", std::int64_t(5)));
    f.add(2, std::vector<float>{1}, Metadata().set("n", std::int64_t(-5)));
    f.add(3, std::vector<float>{2}, Metadata().set("n", lo));
    f.add(4, std::vector<float>{3}, Metadata().set("x", -2.5));
    using V = std::vector<std::uint64_t>;
    CHECK(matching_ids(f, Filter::lt("n", 9.3e18)) == V({1, 2, 3}));       // beyond 2^63: every int is below
    CHECK(matching_ids(f, Filter::gt("n", -9.3e18)) == V({1, 2, 3}));      // beyond -2^63: every int is above
    CHECK(matching_ids(f, Filter::lt("n", 5.5)) == V({1, 2, 3}));          // positive fraction
    CHECK(matching_ids(f, Filter::gt("n", 4.5)) == V({1}));
    CHECK(matching_ids(f, Filter::gt("n", -5.5)) == V({1, 2}));            // negative fraction
    CHECK(matching_ids(f, Filter::lt("n", -4.5)) == V({2, 3}));
    CHECK(matching_ids(f, Filter::eq("n", 5.0)) == V({1}));                // equal, no fraction
    CHECK(matching_ids(f, Filter::eq("n", -9223372036854775808.0)) == V({3}));
    CHECK(matching_ids(f, Filter::gt("x", -3)) == V({4}) && matching_ids(f, Filter::lt("x", -2)) == V({4}));
    CHECK(matching_ids(f, Filter::in("n", {5.0, 7.5})) == V({1}));         // floats against an int field
    CHECK(matching_ids(f, Filter::in("x", {-3, -2})).empty());             // ints against a float field
    CHECK(matching_ids(f, Filter::between("x", -3, -2.0)) == V({4}));     // mixed bounds
    CHECK(matching_ids(f, Filter::between("n", 6, 5.5)).empty());          // low above high (mixed types)
}

TEST(filter, fl63_membership_and_tag_variants) {
    auto f = small_meta_flat(60);
    CHECK(matching_ids(*f, Filter::in("in_stock", {true})) == reference_ids(60, Filter::eq("in_stock", true)));
    CHECK(matching_ids(*f, Filter::in("in_stock", {false})) == reference_ids(60, Filter::eq("in_stock", false)));
    CHECK(matching_ids(*f, Filter::in("in_stock", {true, false})) == reference_ids(60, Filter::exists("in_stock")));
    CHECK(matching_ids(*f, Filter::in("category", {"news", "news", "nope"})) == reference_ids(60, Filter::eq("category", "news")));
    CHECK(matching_ids(*f, Filter::in("category", {"nope"})).empty());
    CHECK(matching_ids(*f, Filter::has_all_tags("tags", {})) == reference_ids(60, Filter::exists("tags")));
    CHECK(matching_ids(*f, Filter::has_all_tags("tags", {"a", "zz"})).empty());
    CHECK(matching_ids(*f, Filter::has_any_tag("tags", {"zz", "yy"})).empty());
    CHECK(matching_ids(*f, Filter::has_any_tag("tags", {"a", "a"})) == reference_ids(60, Filter::has_tag("tags", "a")));
}

TEST(filter, fl64_options_variants) {
    // Filter::all() given explicitly, planner settings on Flat, statistics from range search.
    const MetaFixture& fx = meta_fixture();
    SearchOptions all = with_filter(Filter::all());
    all.planner = PlannerParams{};
    CHECK(same_results(fx.hnsw->search(fx.queries[0], 10, 64, all), fx.hnsw->search(fx.queries[0], 10, 64)));
    CHECK(same_results(fx.flat->search(fx.queries[0], 10, all), fx.flat->search(fx.queries[0], 10)));
    CHECK(fx.hnsw->search_batch(flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 2)), 5, 64, 2, all).size() == 2);
    SearchStats hs, fs;
    SearchOptions o;
    o.stats = &hs;
    o.strategy = Strategy::ForceGraph;
    fx.hnsw->search_range(fx.queries[0], 2.0f, kNoLimit, o);
    CHECK(hs.strategy == Strategy::ForceGraph && hs.reason == PlanReason::Forced);
    o.stats = &fs;
    fx.flat->search_range(fx.queries[0], 2.0f, kNoLimit, o);
    CHECK(fs.reason == PlanReason::FlatIndex && fs.distance_computations == 3000);
    SearchOptions exact = with_filter(Filter::lt("bucket", 10), Strategy::ForceExact);
    const auto r = fx.hnsw->search_range(fx.queries[0], 4.0f, kNoLimit, exact);
    for (const auto& x : r) CHECK(x.id % 100 < 10);
    CHECK(!fx.flat->set_metadata(999999, Metadata().set("year", 1)));
}

TEST(planner, pl17_exact_counts_for_every_filter_kind) {
    auto data = clustered(600, 8, 4, 390);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    for (const char* field : {"category", "tags", "in_stock"}) h.create_payload_index(field);
    auto count = [&](const Filter& flt) {
        SearchStats st;
        SearchOptions o = with_filter(flt);
        o.stats = &st;
        h.search(data[0], 5, 64, o);
        return std::pair<bool, std::size_t>{st.exact_count && st.estimated_matches == reference_ids(600, flt).size(), st.exact_count};
    };
    for (const Filter& flt : {Filter::eq("category", "tech"), Filter::in("category", {"tech", "food"}),
                              Filter::eq("in_stock", true), Filter::eq("in_stock", false), Filter::ne("in_stock", true),
                              Filter::has_tag("tags", "c"), Filter::exists("year"), Filter::between("year", 2010, 2005)})
        CHECK(count(flt).first);
    HnswIndex plain(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));  // no payload index
    for (std::size_t i = 0; i < data.size(); ++i) plain.add(i, data[i], meta_for(i));
    SearchStats st;
    SearchOptions o = with_filter(Filter::in("category", {"tech"}));
    o.stats = &st;
    plain.search(data[0], 5, 64, o);
    CHECK(!st.exact_count && std::fabs(st.selectivity - 0.2) < 0.08);  // no payload index: sampled
    CHECK(h.metadata().value_count(*h.metadata().find_field("category"), 999999) == 0);  // unknown value ID
}

TEST(planner, pl18_plan_search_directly) {
    IdMap ids;
    for (std::uint64_t i = 0; i < 1000; ++i) ids.add(i);
    for (NodeId i = 0; i < 1000; ++i)
        if (i % 50 != 0) ids.mark_deleted(i);  // 20 live slots out of 1,000
    const Eligibility everything{&ids};
    PlanDecision d = plan_search(1000, 0, everything, std::nullopt, 64, Strategy::Auto, PlannerParams{});
    CHECK(d.strategy == Strategy::ForceExact && d.estimated_matches == 0);  // no live vectors
    PlannerParams sparse{0, 0.0, 300, 32.0};  // sample size 300 > live: counted exactly
    d = plan_search(1000, 20, everything, std::nullopt, 64, Strategy::Auto, sparse);
    CHECK(d.exact_count && d.estimated_matches == 20);
    PlannerParams tiny_sample{0, 0.0, 4, 32.0};  // sampling a mostly removed index
    d = plan_search(1000, 20, everything, std::nullopt, 64, Strategy::Auto, tiny_sample);
    CHECK(!d.exact_count && d.selectivity == 1.0 && d.strategy == Strategy::ForceGraph);
    d = plan_search(1000, 20, everything, std::size_t(7), 64, Strategy::ForceGraph, PlannerParams{});
    CHECK(d.reason == PlanReason::Forced && d.exact_count && d.estimated_matches == 7);
}

TEST(metadata, md60_every_type_through_every_operation) {
    MetadataStore m(Schema::dynamic());
    const Metadata full = Metadata().set("i", 7).set("f", 2.5).set("b", true).set("k", "key").set_tags("t", {"x", "y"});
    m.write(0, full);
    m.write(1, Metadata().set("i", 8).set("f", 3).set("b", false).set("k", "other").set_tags("t", {"y"}));
    for (const char* field : {"b", "k", "t"}) m.index_field(field);
    CHECK(m.read(0) == full);
    m.update(1, Metadata().unset("i").unset("never_existed").set("k", "key"));
    CHECK(!m.read(1).get("i") && m.read(1).get("k")->s == "key" && m.read(1).get("f")->f == 3.0);
    m.write(5, full);
    m.move_row(5, 2);  // every type moves
    CHECK(m.read(2) == full && m.read(5).empty());
    m.clear(0);
    m.clear(1);
    CHECK(m.unused_string_count() == 1);  // "other"
    CHECK(m.compact_dictionary() && !m.compact_dictionary());
    CHECK(m.read(2) == full && *m.value_count(*m.find_field("k"), *m.find_string("key")) == 1);
    CHECK(*m.value_count(*m.find_field("b"), 1) == 1 && *m.value_count(*m.find_field("t"), *m.find_string("y")) == 1);
    CHECK(m.current_schema().mode() == SchemaMode::Dynamic && m.current_schema().fields().size() == 5);
    MetadataStore strict(Schema::strict({{"a", FieldType::Int}}));
    CHECK(strict.current_schema().mode() == SchemaMode::Strict);
    CHECK(!(Value::of_int(1) == Value::of_float(1.0)) && Value::of_tags({"a", "b"}) == Value::of_tags({"b", "a", "a"}));
    CHECK_THROWS_AS(std::invalid_argument, m.validate(Metadata().set("", 1)));
}

TEST(batch, bt16_thread_pool_failures) {
    ThreadPool pool(3);
    pool.parallel_for(0, [](std::size_t) { CHECK(false); });  // nothing to do
    CHECK_THROWS_AS(std::runtime_error, pool.parallel_for(50, [](std::size_t) { throw std::runtime_error("every task fails"); }));
    if constexpr (!HNSW_ALLOC_HOOK) return;
    // Running out of memory while starting threads or queuing tasks: clean failure, pool still usable.
    CHECK(sweep_allocation_failures([] { ThreadPool p(4); }, [] {}) > 0);
    std::atomic<std::size_t> sum{0};
    CHECK(sweep_allocation_failures([&] { pool.parallel_for(8, [&](std::size_t i) { sum += i; }); }, [] {}) > 0);
    sum = 0;
    pool.parallel_for(100, [&](std::size_t i) { sum += i; });
    CHECK(sum.load() == 4950);
}

TEST(range, rg16_range_search_next_to_removed_nodes) {
    // Sparse, mostly removed graphs (repair off, M 2-4): later inserts can prune
    // the only links to an earlier vector, making it unreachable for any search,
    // as the M = 2 tests measure. What range search must guarantee is that it
    // finds every live vector reachable from the entry point, which it does by
    // always starting from the entry point as well as from the greedy descent.
    bool complete = true;
    for (unsigned seed = 1; seed <= 150; ++seed) {
        std::mt19937 rng(seed);
        HnswParams p;
        p.M = 2 + seed % 3;
        p.ef_construction = 1 + seed % 4;
        p.repair_on_remove = false;
        p.seed = seed;
        HnswIndex h(2, Metric::L2, p);
        const int n = 20 + int(seed % 30);
        for (int i = 0; i < n; ++i) h.add(std::uint64_t(i), std::vector<float>{portable_uniform(rng), portable_uniform(rng)});
        const std::uint64_t entry = h.storage().ids().external(h.entry_point());
        for (int i = 0; i < n; ++i)
            if (std::uint64_t(i) != entry) h.remove(std::uint64_t(i));
        for (std::uint64_t i = 0; i < 5; ++i) h.add(1000 + i, std::vector<float>{portable_uniform(rng), portable_uniform(rng)});
        const std::vector<float> q{portable_uniform(rng), portable_uniform(rng)};

        // Live vectors reachable from the entry point on level 0 (through removed ones too).
        const Storage& st = h.storage();
        std::vector<bool> seen(st.size(), false);
        std::vector<NodeId> stack{h.entry_point()};
        seen[h.entry_point()] = true;
        while (!stack.empty()) {
            const NodeId a = stack.back();
            stack.pop_back();
            for (NodeId b : st.graph().links(a, 0)) {
                if (b == kEmpty) break;
                if (!seen[b]) {
                    seen[b] = true;
                    stack.push_back(b);
                }
            }
        }
        std::set<std::uint64_t> found;
        for (const auto& r : h.search_range(q, kInf)) found.insert(r.id);
        for (NodeId n2 = 0; n2 < st.size(); ++n2)
            if (seen[n2] && !st.ids().is_deleted(n2)) complete = complete && found.count(st.ids().external(n2)) == 1;
        for (std::uint64_t id : found) complete = complete && h.contains(id);
    }
    CHECK(complete);
}

// ===========================================================================
// Updating vectors (U1-U20 from the test plan, U21-U27 beyond it)
// ===========================================================================

namespace {

/// True if a search for `v` finds `id` stored exactly there (L2 and cosine).
bool found_at(const FlatIndex& f, std::uint64_t id, const std::vector<float>& v) {
    for (const auto& r : f.search(v, 5))
        if (r.id == id && r.distance <= 1e-5f) return true;
    return false;
}
bool found_at(const HnswIndex& h, std::uint64_t id, const std::vector<float>& v) {
    for (const auto& r : h.search(v, 5, 128))
        if (r.id == id && r.distance <= 1e-5f) return true;
    return false;
}

/// HNSW's exact search equals Flat's for every query: both store the same vectors.
bool same_contents(const HnswIndex& h, const FlatIndex& f, const Vectors& queries, std::size_t k = 10) {
    SearchOptions exact;
    exact.strategy = Strategy::ForceExact;
    for (const auto& q : queries)
        if (!same_results(tie_sorted(h.search(q, k, 64, exact)), tie_sorted(f.search(q, k)))) return false;
    return true;
}

}  // namespace

TEST(update, u01_flat_update_moves_the_vector) {
    auto data = clustered(300, 8, 4, 400);
    auto moved = clustered(20, 8, 4, 401);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    auto expected = data;
    for (std::size_t i = 0; i < 20; ++i) {
        CHECK(f.update(i * 7, moved[i]));
        expected[i * 7] = moved[i];
    }
    bool ok = f.size() == 300;
    for (std::size_t i = 0; i < 20; ++i) {
        ok = ok && found_at(f, i * 7, moved[i]);
        for (const auto& r : f.search(data[i * 7], 300)) ok = ok && !(r.id == i * 7 && r.distance <= 1e-5f);  // old position gone
    }
    CHECK(ok);
    for (std::size_t i = 0; i < 10; ++i)
        CHECK(same_results(tie_sorted(f.search(moved[i], 10)),
                           tie_sorted(reference_search(expected, std::vector<bool>(300), moved[i], Metric::L2, 10, 0))));
}

TEST(update, u02_flat_update_unknown_or_removed) {
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) f.add(i, std::vector<float>{float(i), 0});
    f.remove(3);
    const Results before = f.search(std::vector<float>{2.5f, 0}, 10);
    CHECK(!f.update(3, std::vector<float>{9, 9}));
    CHECK(!f.update(99, std::vector<float>{9, 9}));
    CHECK(same_results(f.search(std::vector<float>{2.5f, 0}, 10), before) && f.size() == 9 && !f.contains(3));
    CHECK_THROWS_AS(std::invalid_argument, f.update(99, std::vector<float>{kNaN, 0}));  // input checked first
}

TEST(update, u03_flat_invalid_input_keeps_old_vector) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{1, 2});
    CHECK_THROWS_AS(std::invalid_argument, f.update(1, std::vector<float>{kNaN, 0}));
    CHECK_THROWS_AS(std::invalid_argument, f.update(1, std::vector<float>{kInf, 0}));
    CHECK_THROWS_AS(std::invalid_argument, f.update(1, std::vector<float>{1, 2, 3}));
    CHECK(found_at(f, 1, {1, 2}) && f.size() == 1);
}

TEST(update, u04_flat_update_to_same_vector) {
    auto data = random_vectors(100, 8, 402);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i]);
    const Results before = f.search(data[5], 20);
    for (std::size_t i = 0; i < data.size(); i += 3) CHECK(f.update(i, data[i]));
    CHECK(same_results(f.search(data[5], 20), before));
}

TEST(update, u05_flat_update_normalizes_for_cosine) {
    FlatIndex f(3, Metric::Cosine), fresh(3, Metric::Cosine);
    f.add(1, std::vector<float>{1, 0, 0});
    f.update(1, std::vector<float>{0, 30, 40});  // length 50
    fresh.add(1, std::vector<float>{0, 0.6f, 0.8f});
    const std::vector<float> q{0.2f, 0.5f, 0.1f};
    CHECK(found_at(f, 1, {0, 3, 4}));
    CHECK(std::fabs(f.search(q, 1)[0].distance - fresh.search(q, 1)[0].distance) < 1e-6f);
}

TEST(update, u06_flat_upsert) {
    FlatIndex f(2, Metric::L2);
    CHECK(f.upsert(1, std::vector<float>{0, 0}) && f.size() == 1);    // new: added
    CHECK(!f.upsert(1, std::vector<float>{5, 5}) && f.size() == 1);   // existing: replaced
    CHECK(found_at(f, 1, {5, 5}) && !found_at(f, 1, {0, 0}));
    CHECK(f.upsert(2, std::vector<float>{1, 1}) && f.size() == 2);
}

TEST(update, u07_hnsw_update_moves_the_vector) {
    auto data = clustered(2000, 16, 10, 403);
    auto moved = clustered(50, 16, 10, 404);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::size_t i = 0; i < 50; ++i) {
        CHECK(h.update(i * 31, moved[i]));
        f.update(i * 31, moved[i]);
    }
    bool ok = h.size() == 2000;
    for (std::size_t i = 0; i < 50; ++i) {
        ok = ok && found_at(h, i * 31, moved[i]);
        for (const auto& r : h.search(data[i * 31], 50, 128)) ok = ok && !(r.id == i * 31 && r.distance <= 1e-5f);
    }
    CHECK(ok);
    CHECK(recall(h, f, moved, 10, 64) >= 0.95);
    check_graph(h, 0.99);
}

TEST(update, u08_hnsw_update_capacity_bounded) {
    auto data = clustered(500, 8, 4, 405);
    auto moved = clustered(10, 8, 4, 406);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    CHECK(h.update(7, moved[0]));
    CHECK(h.size() == 500 && h.capacity() == 501 && h.deleted_count() == 1);  // one extra slot...
    for (std::size_t i = 1; i < 10; ++i) CHECK(h.update(i * 11, moved[i]));
    CHECK(h.size() == 500 && h.capacity() == 501 && h.deleted_count() == 1);  // ...reused by every later update
    h.add(9999, moved[0]);
    CHECK(h.capacity() == 501 && h.deleted_count() == 0);  // and by inserts
}

TEST(update, u09_hnsw_update_unknown_or_removed) {
    auto data = clustered(300, 8, 4, 407);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    h.remove(5);
    const Results before = h.search(data[0], 10);
    const std::size_t cap = h.capacity(), del = h.deleted_count();
    const NodeId entry = h.entry_point();
    CHECK(!h.update(5, data[1]) && !h.update(12345, data[1]));
    CHECK(same_results(h.search(data[0], 10), before) && h.capacity() == cap && h.deleted_count() == del);
    CHECK(h.entry_point() == entry && h.size() == 299);
}

TEST(update, u10_hnsw_invalid_input_keeps_old_vector) {
    auto data = clustered(300, 8, 4, 408);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const NodeId entry = h.entry_point();
    const int top = h.max_level();
    const std::size_t cap = h.capacity();
    std::vector<float> nan = data[3], short_vec(7, 0.0f);
    nan[2] = kNaN;
    CHECK_THROWS_AS(std::invalid_argument, h.update(3, nan));
    CHECK_THROWS_AS(std::invalid_argument, h.update(3, short_vec));
    CHECK(found_at(h, 3, data[3]) && h.entry_point() == entry && h.max_level() == top && h.capacity() == cap);
}

TEST(update, u11_update_the_entry_point) {
    auto data = clustered(800, 8, 4, 409);
    auto far = clustered(30, 8, 4, 410);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    bool ok = true;
    for (std::size_t round = 0; round < 30; ++round) {
        const std::uint64_t entry_id = h.storage().ids().external(h.entry_point());
        CHECK(h.update(entry_id, far[round]));
        const NodeId e = h.entry_point();
        ok = ok && e != kEmpty && !h.storage().ids().is_deleted(e) && h.storage().graph().level(e) == h.max_level();
        ok = ok && found_at(h, entry_id, far[round]);
    }
    CHECK(ok);
    check_graph(h, 0.99);
}

TEST(update, u12_update_every_vector) {
    auto data = clustered(2000, 16, 10, 411);
    auto moved = clustered(2000, 16, 10, 412);
    auto queries = clustered(40, 16, 10, 413);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::size_t i = 0; i < 2000; ++i) {
        h.update(i, moved[i]);
        f.update(i, moved[i]);
    }
    CHECK(h.size() == 2000 && h.capacity() <= 2001);
    CHECK(recall(h, f, queries, 10, 64) >= 0.95);
    CHECK(same_contents(h, f, queries));
    check_graph(h, 0.99);
}

TEST(update, u13_same_id_updated_1000_times) {
    auto data = clustered(400, 8, 4, 414);
    auto pool = clustered(1000, 8, 4, 415);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    std::size_t slots_after_100 = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        h.update(42, pool[i]);
        if (i == 100) slots_after_100 = h.storage().graph().upper_slots_allocated();
    }
    CHECK(h.size() == 400 && h.capacity() <= 401 && h.deleted_count() <= 1);
    CHECK(h.storage().graph().upper_slots_allocated() <= slots_after_100 + 64);  // link memory stable
    CHECK(found_at(h, 42, pool[999]));
    check_graph(h, 0.99);
}

TEST(update, u14_small_and_large_moves) {
    auto data = clustered(1500, 16, 10, 416);
    auto far = clustered(300, 16, 10, 417);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    std::mt19937 rng(418);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    Vectors queries;
    for (std::size_t i = 0; i < 300; ++i) {
        auto nudged = data[i];
        for (float& x : nudged) x += noise(rng);  // small move
        h.update(i, nudged);
        f.update(i, nudged);
        h.update(1000 + i, far[i]);  // large move
        f.update(1000 + i, far[i]);
        if (i % 15 == 0) {
            queries.push_back(nudged);
            queries.push_back(far[i]);
        }
    }
    bool found = true;
    for (std::size_t i = 0; i < 300; i += 7) found = found && found_at(h, 1000 + i, far[i]);
    CHECK(found);
    CHECK(recall(h, f, queries, 10, 64) >= 0.90);
}

TEST(update, u15_update_to_a_copy_of_another_vector) {
    auto data = clustered(300, 8, 4, 419);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    h.update(10, data[200]);
    f.update(10, data[200]);
    std::set<std::uint64_t> at_zero_h, at_zero_f;
    for (const auto& r : h.search(data[200], 5, 128)) if (r.distance <= 1e-6f) at_zero_h.insert(r.id);
    for (const auto& r : f.search(data[200], 5)) if (r.distance <= 1e-6f) at_zero_f.insert(r.id);
    CHECK(at_zero_h == std::set<std::uint64_t>({10, 200}) && at_zero_f == at_zero_h);
}

TEST(update, u16_index_with_one_vector) {
    HnswIndex h(2, Metric::L2);
    h.add(1, std::vector<float>{0, 0});
    CHECK(h.update(1, std::vector<float>{5, 5}));
    CHECK(h.size() == 1 && h.entry_point() != kEmpty && found_at(h, 1, {5, 5}));
    CHECK(h.update(1, std::vector<float>{-3, 1}) && found_at(h, 1, {-3, 1}));
    for (std::uint64_t i = 2; i < 50; ++i) h.add(i, std::vector<float>{float(i), float(i % 7)});
    CHECK(found_at(h, 1, {-3, 1}) && h.size() == 49);
    check_graph(h, 1.0);
}

TEST(update, u17_hnsw_upsert_matches_flat) {
    auto pool = clustered(3000, 8, 6, 420);
    auto queries = clustered(20, 8, 6, 421);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    std::mt19937 rng(422);
    bool same_returns = true;
    for (std::size_t i = 0; i < 3000; ++i) {
        const std::uint64_t id = rng() % 800;  // many repeats: about 3 upserts per ID
        same_returns = same_returns && h.upsert(id, pool[i]) == f.upsert(id, pool[i]);
    }
    CHECK(same_returns && h.size() == f.size());
    CHECK(same_contents(h, f, queries));
    CHECK(recall(h, f, queries, 10, 64) >= 0.95);
}

TEST(update, u18_every_metric) {
    auto data = clustered(600, 12, 6, 423);
    auto moved = clustered(150, 12, 6, 424);
    for (Metric m : kMetrics) {
        HnswIndex h(12, m);
        FlatIndex f(12, m);
        fill(h, f, data);
        for (std::size_t i = 0; i < 150; ++i) {
            h.update(i * 4, moved[i]);
            f.update(i * 4, moved[i]);
        }
        CHECK(same_contents(h, f, moved));
        CHECK(recall(h, f, moved, 10, 64) >= (m == Metric::InnerProduct ? 0.80 : 0.95));
    }
}

TEST(update, u19_reproducible) {
    auto data = clustered(700, 8, 4, 425);
    auto moved = clustered(300, 8, 4, 426);
    HnswIndex a(8, Metric::Cosine), b(8, Metric::Cosine);
    for (HnswIndex* h : {&a, &b}) {
        for (std::size_t i = 0; i < data.size(); ++i) h->add(i, data[i]);
        for (std::size_t i = 0; i < 300; ++i) {
            h->update(i * 2, moved[i]);
            if (i % 5 == 0) h->remove(i * 2 + 1);
            if (i % 7 == 0) h->upsert(5000 + i, moved[i]);
        }
    }
    bool same = a.capacity() == b.capacity() && a.entry_point() == b.entry_point();
    for (std::size_t i = 0; i < 30; ++i) same = same && same_results(a.search(moved[i], 10), b.search(moved[i], 10));
    CHECK(same);
}

TEST(update, u20_oom_old_vector_survives) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 8, 4, 427);
    auto moved = clustered(2, 8, 4, 428);
    for (int with_free_slot = 0; with_free_slot < 2; ++with_free_slot) {
        HnswIndex h(8, Metric::L2);
        for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], Metadata().set("tag", "v" + std::to_string(i)));
        if (with_free_slot) h.remove(299);  // the update then reuses a free slot instead of appending
        const long points = sweep_allocation_failures([&] { h.update(5, moved[0]); }, [&] {
            CHECK(h.contains(5) && found_at(h, 5, data[5]) && !found_at(h, 5, moved[0]));
            CHECK(h.get_metadata(5)->get("tag")->s == "v5" && h.size() == (with_free_slot ? 299u : 300u));
            check_graph(h, 0.99);
        });
        CHECK(points > 0);
        CHECK(found_at(h, 5, moved[0]) && h.get_metadata(5)->get("tag")->s == "v5");
    }
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i], Metadata().set("tag", "v" + std::to_string(i)));
    const Metadata replacement = Metadata().set("tag", "brand_new").set("extra", 1);
    const long points = sweep_allocation_failures([&] { f.update(5, moved[1], replacement); }, [&] {
        CHECK(found_at(f, 5, data[5]) && *f.get_metadata(5) == Metadata().set("tag", "v5"));
        CHECK(!f.metadata().find_field("extra").has_value());
    });
    CHECK(points > 0);
    CHECK(found_at(f, 5, moved[1]) && *f.get_metadata(5) == replacement);
}

TEST(update, u21_update_keeps_metadata) {
    auto data = clustered(400, 8, 4, 429);
    auto moved = clustered(40, 8, 4, 430);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    FlatIndex f(8, Metric::L2, Schema::strict(search_fields()));
    h.create_payload_index("category");
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const std::size_t news = *h.metadata().value_count(0, *h.metadata().find_string("news"));
    for (std::size_t i = 0; i < 40; ++i) {
        h.update(i * 10, moved[i]);
        f.update(i * 10, moved[i]);
    }
    bool kept = true;
    for (std::size_t i = 0; i < 40; ++i) kept = kept && *h.get_metadata(i * 10) == meta_for(i * 10) && *f.get_metadata(i * 10) == meta_for(i * 10);
    CHECK(kept && *h.metadata().value_count(0, *h.metadata().find_string("news")) == news);
    const SearchOptions o = with_filter(Filter::eq("category", "news"));  // ids divisible by 5: every updated one
    for (std::size_t i = 0; i < 40; i += 4) {
        const auto r = h.search(moved[i], 1, 64, o);
        CHECK(r.size() == 1 && r[0].id == i * 10 && r[0].distance <= 1e-5f);
    }
}

TEST(update, u22_update_with_metadata_replaces_all_fields) {
    for (int which = 0; which < 2; ++which) {
        HnswIndex h(2, Metric::L2);
        FlatIndex f(2, Metric::L2);
        const Metadata old_md = Metadata().set("a", 1).set("b", "x").set_tags("t", {"p", "q"});
        const Metadata new_md = Metadata().set("b", "y");
        which ? h.add(1, std::vector<float>{0, 0}, old_md) : f.add(1, std::vector<float>{0, 0}, old_md);
        which ? h.add(2, std::vector<float>{9, 9}) : f.add(2, std::vector<float>{9, 9});
        CHECK(which ? h.update(1, std::vector<float>{3, 3}, new_md) : f.update(1, std::vector<float>{3, 3}, new_md));
        const auto md = which ? h.get_metadata(1) : f.get_metadata(1);
        CHECK(*md == new_md);  // a and t cleared
        CHECK(which ? found_at(h, 1, {3, 3}) : found_at(f, 1, {3, 3}));
        const MetadataStore& m = which ? h.metadata() : f.metadata();
        CHECK(m.unused_string_count() == 3);  // "x", "p" and "q" no longer used
    }
}

TEST(update, u23_upsert_with_and_without_metadata) {
    for (int which = 0; which < 2; ++which) {
        HnswIndex h(2, Metric::L2);
        FlatIndex f(2, Metric::L2);
        auto upsert = [&](std::uint64_t id, std::vector<float> v, const Metadata* md) {
            if (which) return md ? h.upsert(id, v, *md) : h.upsert(id, v);
            return md ? f.upsert(id, v, *md) : f.upsert(id, v);
        };
        auto meta = [&](std::uint64_t id) { return which ? h.get_metadata(id) : f.get_metadata(id); };
        const Metadata first = Metadata().set("n", 1), second = Metadata().set("m", 2);
        CHECK(upsert(1, {0, 0}, &first) && *meta(1) == first);     // added with metadata
        CHECK(!upsert(1, {1, 1}, nullptr) && *meta(1) == first);   // replaced, metadata kept
        CHECK(!upsert(1, {2, 2}, &second) && *meta(1) == second);  // replaced with new metadata
        CHECK(upsert(2, {5, 5}, nullptr) && meta(2)->empty());     // added without metadata
    }
}

TEST(update, u24_invalid_metadata_changes_nothing) {
    auto data = clustered(200, 8, 4, 431);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    FlatIndex f(8, Metric::L2, Schema::strict(search_fields()));
    fill(h, f, data);
    for (const Metadata& bad : {Metadata().set("unknown", 1), Metadata().set("year", "text"), Metadata().set("price", double(kNaN))}) {
        CHECK_THROWS_AS(std::invalid_argument, h.update(3, data[100], bad));
        CHECK_THROWS_AS(std::invalid_argument, f.update(3, data[100], bad));
        CHECK_THROWS_AS(std::invalid_argument, h.upsert(3, data[100], bad));
    }
    CHECK(found_at(h, 3, data[3]) && found_at(f, 3, data[3]) && h.capacity() == 200);
}

TEST(update, u25_unmapped_slot_primitives) {
    IdMap m;
    m.add(10);  // slot 0
    m.add(11);  // slot 1
    const NodeId s = m.add_unmapped(10);  // a second slot holding ID 10
    CHECK(s == 2 && *m.find(10) == 0 && m.external(2) == 10 && !m.is_deleted(2));
    m.repoint(10, 2);
    m.retire(0);
    CHECK(*m.find(10) == 2 && m.is_free(0) && !m.is_free(2));
    CHECK_THROWS_AS(std::logic_error, m.occupy_unmapped(1, 99));  // slot 1 is live
    m.occupy_unmapped(0, 11);  // slot 0 now holds ID 11 too, unfindable
    CHECK(*m.find(11) == 1 && !m.is_free(0));
    m.retire(0);
    CHECK(m.is_free(0));

    Storage st(2);
    st.insert(1, std::vector<float>{0, 0}, 0);
    CHECK_THROWS_AS(std::invalid_argument, st.insert_unmapped(1, std::vector<float>{0}, 0));
    CHECK_THROWS_AS(std::invalid_argument, st.insert_unmapped(1, std::vector<float>{0, 0}, 256));
    CHECK_THROWS_AS(std::logic_error, st.insert_into_unmapped(0, 1, std::vector<float>{0, 0}, 0));  // live slot
    CHECK(st.size() == 1);
    const NodeId n = st.insert_unmapped(1, std::vector<float>{4, 4}, 2);
    CHECK(n == 1 && *st.ids().find(1) == 0 && st.graph().level(1) == 2 && st.vectors().get(1)[0] == 4.0f);
}

TEST(update, u26_update_without_graph_repair) {
    auto data = clustered(1500, 16, 8, 432);
    auto moved = clustered(500, 16, 8, 433);
    HnswParams p;
    p.repair_on_remove = false;
    HnswIndex h(16, Metric::L2, p);
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    for (std::size_t i = 0; i < 500; ++i) {
        h.update(i * 3, moved[i]);
        f.update(i * 3, moved[i]);
    }
    CHECK(same_contents(h, f, moved));
    CHECK(recall(h, f, moved, 10, 64) >= 0.90);
    check_graph(h, 0.98, 0.10);
}

TEST(update, u27_mixed_operations_against_reference) {
    // 5,000 random adds, removes, updates, upserts and metadata changes on both
    // indexes, checked against each other and a reference map throughout.
    auto pool = clustered(6000, 8, 8, 434);
    auto queries = clustered(15, 8, 8, 435);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    std::map<std::uint64_t, std::int64_t> ref;  // id -> "v" field
    std::mt19937 rng(436);
    std::size_t next = 0, peak = 0;
    bool ok = true;
    for (int step = 0; step < 5000 && next < pool.size(); ++step) {
        const unsigned op = static_cast<unsigned>(rng() % 6);
        const std::uint64_t id = rng() % 1200;
        const auto& v = pool[next++];
        if (op == 0) {
            if (!ref.count(id)) {
                h.add(id, v, Metadata().set("v", std::int64_t(step)));
                f.add(id, v, Metadata().set("v", std::int64_t(step)));
                ref[id] = step;
            }
        } else if (op == 1) {
            ok = ok && h.remove(id) == f.remove(id);
            ref.erase(id);
        } else if (op == 2 || op == 3) {
            ok = ok && h.update(id, v) == f.update(id, v);  // metadata kept
        } else if (op == 4) {
            const bool added = h.upsert(id, v, Metadata().set("v", std::int64_t(step)));
            ok = ok && added == f.upsert(id, v, Metadata().set("v", std::int64_t(step))) && added == !ref.count(id);
            ref[id] = step;
        } else if (ref.count(id)) {
            h.set_metadata(id, Metadata().set("v", std::int64_t(-step)));
            f.set_metadata(id, Metadata().set("v", std::int64_t(-step)));
            ref[id] = -step;
        }
        peak = std::max(peak, ref.size());
        if (step % 500 == 0) ok = ok && same_contents(h, f, queries);
    }
    for (const auto& [id, val] : ref) ok = ok && h.get_metadata(id)->get("v")->i == val && f.get_metadata(id)->get("v")->i == val;
    CHECK(ok && h.size() == ref.size() && f.size() == ref.size());
    CHECK(h.capacity() <= peak + 1);
    CHECK(recall(h, f, queries, 10, 64) >= 0.90);
    check_graph(h, 0.99);
}

// ===========================================================================
// Stress: limits, memory growth under long churn, and fixes from the load audit
// ===========================================================================

TEST(stress, st01_deep_filter_rejected_not_crashing) {
    // Chaining && in a loop used to overflow the stack (found by the load audit).
    Filter flt = Filter::eq("n", 1);
    bool threw = false;
    try {
        for (int i = 0; i < 200000; ++i) flt = flt && Filter::eq("n", 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw && flt.root().depth == Filter::kMaxDepth);
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("n", 1));
    CHECK(f.search(std::vector<float>{0}, 1, with_filter(flt)).size() == 1);  // the deepest allowed filter works
    CHECK_THROWS_AS(std::invalid_argument, !flt);  // one level too deep
}

TEST(stress, st02_wide_balanced_filter) {
    // 4,096 conditions as a balanced tree (depth 13) evaluate correctly.
    std::vector<Filter> level;
    for (int i = 0; i < 4096; ++i) level.push_back(Filter::eq("n", i));
    while (level.size() > 1) {
        std::vector<Filter> next;
        for (std::size_t i = 0; i < level.size(); i += 2) next.push_back(level[i] || level[i + 1]);
        level.swap(next);
    }
    FlatIndex f(1, Metric::L2);
    for (std::int64_t i = 0; i < 5000; ++i) f.add(std::uint64_t(i), std::vector<float>{float(i)}, Metadata().set("n", i));
    CHECK(f.search(std::vector<float>{0}, kHuge, with_filter(level[0])).size() == 4096);
}

TEST(stress, st03_dictionary_bounded_under_churn) {
    // Unique strings with few live vectors used to grow the dictionary forever.
    for (int which = 0; which < 2; ++which) {
        FlatIndex f(1, Metric::L2);
        HnswIndex h(1, Metric::L2);
        std::size_t peak = 0;
        for (std::uint64_t i = 0; i < 100000; ++i) {
            const Metadata md = Metadata().set("session", "s" + std::to_string(i)).set_tags("t", {"u" + std::to_string(i)});
            which == 0 ? f.add(i, std::vector<float>{float(i % 100)}, md) : h.add(i, std::vector<float>{float(i % 100)}, md);
            if (i >= 10) which == 0 ? f.remove(i - 10) : h.remove(i - 10);
            peak = std::max(peak, (which == 0 ? f.metadata() : h.metadata()).dictionary_size());
            if (which == 1 && i == 20000) break;  // HNSW is slower; 20,000 steps suffice
        }
        CHECK(peak <= 4200);  // 2 strings per live vector + at most ~2,048 unused before reclaiming
        const MetadataStore& m = which == 0 ? f.metadata() : h.metadata();
        const std::uint64_t last = which == 0 ? 99999 : 20000;
        const auto md = which == 0 ? f.get_metadata(last) : h.get_metadata(last);
        CHECK(md && md->get("session")->s == "s" + std::to_string(last));  // still correct after renumbering
        CHECK(m.find_string("s0") == std::nullopt);                           // long-gone strings reclaimed
    }
}

TEST(stress, st04_dictionary_compaction_keeps_filters_and_counts) {
    FlatIndex f(1, Metric::L2, Schema::strict(search_fields()));
    f.create_payload_index("category");
    f.create_payload_index("tags");
    for (std::uint64_t i = 0; i < 3000; ++i)
        f.add(i, std::vector<float>{float(i)}, Metadata().set("category", "c" + std::to_string(i)).set_tags("tags", {"t" + std::to_string(i % 7)}));
    for (std::uint64_t i = 0; i < 3000; ++i)
        if (i % 10 != 0) f.remove(i);  // 300 left; 2,700 category strings unused
    CHECK(f.metadata().unused_string_count() == 2700);
    f.compact();  // explicit compaction
    const MetadataStore& m = f.metadata();
    CHECK(m.unused_string_count() == 0 && m.dictionary_size() == 300 + 7);
    CHECK(matching_ids(f, Filter::eq("category", "c1230")) == std::vector<std::uint64_t>{1230});
    CHECK(matching_ids(f, Filter::eq("category", "c1231")).empty());
    CHECK(m.value_count(*m.find_field("category"), *m.find_string("c1230")) == 1);
    std::size_t t3 = 0;
    for (std::uint64_t i = 0; i < 3000; i += 10) t3 += i % 7 == 3;
    CHECK(m.value_count(*m.find_field("tags"), *m.find_string("t3")) == t3);
    CHECK(matching_ids(f, Filter::has_tag("tags", "t3")).size() == t3);
}

TEST(stress, st05_thread_count_is_capped) {
    const Fixture& fx = fixture(Metric::L2);
    const auto buffer = flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 4));
    const auto h = fx.hnsw->search_batch(buffer, 5, 64, 100000);  // used to throw: too many threads
    const auto f = fx.flat->search_batch(buffer, 5, 100000);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK(same_results(h[i], fx.hnsw->search(fx.queries[i], 5, 64)) && same_results(f[i], fx.flat->search(fx.queries[i], 5)));
    CHECK(effective_threads(100000) <= std::max<std::size_t>(64, 4 * std::max(1u, std::thread::hardware_concurrency())));
    CHECK(effective_threads(0) >= 1 && effective_threads(3) == 3);
}

TEST(stress, st06_upper_blocks_recycled_under_churn) {
    // Reusing a slot at a higher level used to abandon its old link block.
    auto data = clustered(500, 4, 4, 380);
    auto pool = clustered(30000, 4, 4, 381);
    HnswIndex h(4, Metric::L2, HnswParams{4, 32, 5});  // M = 4: levels vary a lot
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    std::vector<std::uint64_t> live(500);
    std::iota(live.begin(), live.end(), 0);
    std::mt19937 rng(382);
    std::size_t next = 0, after_warmup = 0;
    for (int step = 0; step < 30000; ++step) {
        const std::size_t pos = rng() % live.size();
        h.remove(live[pos]);
        h.add(100000 + next, pool[next]);
        live[pos] = 100000 + next++;
        if (step == 10000) after_warmup = h.storage().graph().upper_slots_allocated();
    }
    const std::size_t final_slots = h.storage().graph().upper_slots_allocated();
    CHECK(final_slots <= after_warmup + after_warmup / 5);  // essentially flat, not growing with churn
    CHECK(h.storage().graph().recycled_upper_blocks() > 0);
    check_graph(h, 0.95);
}

TEST(stress, st07_compact_releases_visited_lists) {
    auto data = clustered(2000, 8, 4, 383);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    h.search(data[0], 5);
    CHECK(h.pooled_visited_lists() >= 1);
    for (std::uint64_t i = 0; i < 2000; i += 2) h.remove(i);
    h.compact();
    CHECK(h.pooled_visited_lists() == 0);
    CHECK(h.search(data[1], 5)[0].id == 1);  // searches work and create a right-sized list
    CHECK(h.pooled_visited_lists() == 1);
}

TEST(stress, st08_unsigned_values_beyond_int64_rejected) {
    const std::uint64_t too_big = std::uint64_t(1) << 63;
    CHECK_THROWS_AS(std::invalid_argument, Metadata().set("n", too_big));
    CHECK_THROWS_AS(std::invalid_argument, Filter::eq("n", too_big));
    CHECK_NOTHROW(Metadata().set("n", too_big - 1));
    FlatIndex f(1, Metric::L2);
    f.add(1, std::vector<float>{0}, Metadata().set("n", too_big - 1));
    CHECK(f.get_metadata(1)->get("n")->i == std::numeric_limits<std::int64_t>::max());
}

TEST(stress, st09_large_in_list) {
    FlatIndex f(1, Metric::L2);
    for (std::uint64_t i = 0; i < 20000; ++i) f.add(i, std::vector<float>{float(i)}, Metadata().set("k", "v" + std::to_string(i)));
    std::vector<std::string> wanted;
    for (int i = 0; i < 10000; ++i) wanted.push_back("v" + std::to_string(i * 2));
    const auto ids = matching_ids(f, Filter::in("k", wanted));
    bool ok = ids.size() == 10000;
    for (std::size_t i = 0; ok && i < ids.size(); ++i) ok = ids[i] == i * 2;
    CHECK(ok);
}

TEST(stress, st10_long_churn_memory_bounded) {
    // 60,000 mixed operations on ~1,000 live vectors with metadata: slots,
    // dictionary and link blocks all stay bounded, and results stay correct.
    auto pool = clustered(40000, 8, 8, 384);
    auto queries = clustered(10, 8, 8, 385);
    HnswIndex h(8, Metric::L2, HnswParams{8, 64, 3});
    FlatIndex f(8, Metric::L2);
    std::vector<std::uint64_t> live;
    std::mt19937 rng(386);
    std::size_t next = 0, peak_live = 0, max_dictionary = 0;
    for (int step = 0; step < 60000 && next < pool.size(); ++step) {
        if (live.size() < 1000 || rng() % 2 == 0) {
            const Metadata md = Metadata().set("user", "u" + std::to_string(next)).set("group", std::int64_t(next % 5));
            h.add(next, pool[next], md);
            f.add(next, pool[next], md);
            live.push_back(next++);
        } else {
            const std::size_t pos = rng() % live.size();
            h.remove(live[pos]);
            f.remove(live[pos]);
            live[pos] = live.back();
            live.pop_back();
        }
        peak_live = std::max(peak_live, live.size());
        max_dictionary = std::max(max_dictionary, h.metadata().dictionary_size());
    }
    CHECK(h.capacity() <= peak_live && f.capacity() == live.size());
    CHECK(max_dictionary <= peak_live + 2100);
    const SearchOptions o = with_filter(Filter::eq("group", 2));
    CHECK(filtered_recall(h, f, queries, 10, 64, o) >= 0.90);
    check_graph(h, 0.99);
}

TEST(stress, st11_oom_during_dictionary_compaction) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    FlatIndex f(1, Metric::L2);
    for (std::uint64_t i = 0; i < 3000; ++i) f.add(i, std::vector<float>{float(i)}, Metadata().set("k", "s" + std::to_string(i)));
    for (std::uint64_t i = 0; i < 2900; ++i) f.remove(i);  // 2,900 unused strings: the next write compacts
    const std::size_t before = f.metadata().dictionary_size();
    const long points = sweep_allocation_failures([&] { f.add(5000, std::vector<float>{0}, Metadata().set("k", "fresh")); }, [&] {
        CHECK(!f.contains(5000) && f.metadata().dictionary_size() <= before && f.size() == 100);
        CHECK(matching_ids(f, Filter::eq("k", "s2950")) == std::vector<std::uint64_t>{2950});
    });
    CHECK(points > 0);
    CHECK(f.metadata().unused_string_count() == 0 && f.metadata().dictionary_size() == 101);
    CHECK(matching_ids(f, Filter::eq("k", "fresh")) == std::vector<std::uint64_t>{5000});
}

TEST(stress, st12_oom_upper_block_recycling) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    GraphStorage g(4);
    std::vector<NodeId> nodes;
    for (int i = 0; i < 6; ++i) nodes.push_back(g.add_node(i % 3));
    CHECK(sweep_allocation_failures([&] { g.reset_node(nodes[1], 5); },
                                    [&] { CHECK(g.level(nodes[1]) == 1 && g.recycled_upper_blocks() == 0); }) > 0);
    CHECK(g.level(nodes[1]) == 5 && g.recycled_upper_blocks() == 1);  // its old 1-level block is recycled
    g.reset_node(nodes[0], 1);  // takes the recycled block: no new arena slots
    const std::size_t slots = g.upper_slots_allocated();
    CHECK(g.recycled_upper_blocks() == 0 && g.upper_slots_allocated() == slots);
}

// ===========================================================================
// End to end
// ===========================================================================

namespace {

/// Add 1000, remove 300, add 600 more, checking recall, results and graph
/// after every step. Inner product is harder for HNSW (not a true distance;
/// the reference hnswlib reaches 0.887 / 0.900 / 0.853 on this data, this
/// index 0.898 / 0.905 / 0.860), so it gets lower thresholds.
void lifecycle(Metric m) {
    const double min_recall = (m == Metric::InnerProduct) ? 0.80 : 0.95;
    const double min_reach = (m == Metric::InnerProduct) ? 0.97 : 0.99;
    const std::size_t dim = 24;
    auto data = clustered(1600, dim, 12, 90);
    auto queries = clustered(60, dim, 12, 91);
    HnswIndex h(dim, m);
    FlatIndex f(dim, m);

    for (std::size_t i = 0; i < 1000; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(h.size() == 1000 && f.size() == 1000);
    CHECK(recall(h, f, queries, 10, 100) >= min_recall);

    for (std::size_t i = 0; i < 1000; i += 3)
        if (h.size() > 700) {
            CHECK(h.remove(i));
            CHECK(f.remove(i));
        }
    CHECK(h.size() == 700 && f.size() == 700);
    CHECK(recall(h, f, queries, 10, 100) >= min_recall);
    for (const auto& q : queries) check_results(h, h.search(q, 10, 100));

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
    check_graph(h, min_reach);
}

}  // namespace

TEST(e2e, lifecycle_l2) { lifecycle(Metric::L2); }
TEST(e2e, lifecycle_inner_product) { lifecycle(Metric::InnerProduct); }
TEST(e2e, lifecycle_cosine) { lifecycle(Metric::Cosine); }

TEST(e2e, larger_index_recall) {
    auto data = clustered(6000, 48, 30, 92);
    auto queries = clustered(100, 48, 30, 93);
    HnswIndex h(48, Metric::L2, HnswParams{16, 100, 3});
    FlatIndex f(48, Metric::L2);
    fill(h, f, data);
    CHECK(recall(h, f, queries, 10, 128) >= 0.95);
    check_graph(h, 0.99);
}

TEST(e2e, random_operations_against_flat) {
    // A long random mix of adds, removes and searches. FlatIndex is the oracle:
    // sizes must always agree, results must be valid, and recall must stay high.
    auto pool = clustered(4000, 16, 10, 94);
    std::mt19937 rng(95);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    std::vector<std::uint64_t> live;
    std::size_t next = 0, searches = 0;
    double recall_sum = 0.0;
    bool sizes_agree = true;
    for (int step = 0; step < 5000 && next < pool.size(); ++step) {
        const auto op = static_cast<unsigned>(rng() % 10);
        if (op < 5 || live.size() < 20) {  // add
            h.add(next, pool[next]);
            f.add(next, pool[next]);
            live.push_back(next++);
        } else if (op < 7) {  // remove a random live vector
            const std::size_t pos = rng() % live.size();
            const std::uint64_t id = live[pos];
            live[pos] = live.back();
            live.pop_back();
            if (!h.remove(id) || !f.remove(id)) sizes_agree = false;
        } else {  // search with a random stored vector as the query
            const auto& q = pool[rng() % next];
            auto hr = h.search(q, 10, 100);
            auto fr = f.search(q, 10);
            check_results(h, hr);
            std::set<std::uint64_t> truth;
            for (const auto& r : fr) truth.insert(r.id);
            std::size_t hits = 0;
            for (const auto& r : hr) hits += truth.count(r.id);
            recall_sum += truth.empty() ? 1.0 : double(hits) / double(truth.size());
            ++searches;
        }
        if (h.size() != live.size() || f.size() != live.size()) sizes_agree = false;
    }
    CHECK(sizes_agree);
    CHECK(searches > 500);
    CHECK(recall_sum / double(searches) >= 0.95);
    check_graph(h, 0.99);
}

TEST(e2e, all_metrics_same_data) {
    // The same data in three indexes: each metric gives its own valid ranking.
    auto data = clustered(800, 16, 8, 96);
    auto queries = clustered(30, 16, 8, 97);
    for (Metric m : kMetrics) {
        HnswIndex h(16, m);
        FlatIndex f(16, m);
        fill(h, f, data);
        CHECK(recall(h, f, queries, 10, 128) >= (m == Metric::InnerProduct ? 0.85 : 0.95));
        for (const auto& q : queries) check_results(h, h.search(q, 10, 64));
    }
}

// ===========================================================================
// Runner
// ===========================================================================

namespace {

void print_usage() {
    std::printf(
        "Usage: test_comprehensive [options] [patterns...]\n"
        "\n"
        "  (no arguments)      run every test\n"
        "  patterns            run tests whose \"group.name\" contains any pattern\n"
        "  --group <name>      run only this group (repeatable)\n"
        "  --exclude <text>    skip tests whose \"group.name\" contains text (repeatable)\n"
        "  --list              list the selected tests without running them\n"
        "  --fail-fast         stop after the first failing test\n"
        "  --help              show this help\n"
        "\n"
        "Groups: layer1, layer2, helpers, flat, hnsw, robustness, deletion, concurrency,\n"
        "        metadata, filter, planner, payload, batch, range, search_e2e, update, stress, e2e\n"
        "Examples:\n"
        "  test_comprehensive --group hnsw\n"
        "  test_comprehensive recall\n"
        "  test_comprehensive flat.k_zero\n"
        "  test_comprehensive --exclude e2e --exclude concurrency\n");
}

bool contains_text(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> patterns, groups, excludes;
    bool list_only = false, fail_fast = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--list") {
            list_only = true;
        } else if (arg == "--fail-fast") {
            fail_fast = true;
        } else if (arg == "--group" || arg == "--exclude") {
            if (i + 1 >= argc) {
                std::printf("Missing value after %s\n\n", arg.c_str());
                print_usage();
                return 2;
            }
            (arg == "--group" ? groups : excludes).push_back(argv[++i]);
        } else if (!arg.empty() && arg[0] == '-') {
            std::printf("Unknown option: %s\n\n", arg.c_str());
            print_usage();
            return 2;
        } else {
            patterns.push_back(arg);
        }
    }

    // Select tests.
    std::vector<const TestCase*> selected;
    for (const TestCase& t : registry()) {
        const std::string full = t.full_name();
        if (!groups.empty() && std::find(groups.begin(), groups.end(), t.group) == groups.end()) continue;
        if (!patterns.empty() && std::none_of(patterns.begin(), patterns.end(),
                                              [&](const std::string& p) { return contains_text(full, p); }))
            continue;
        if (std::any_of(excludes.begin(), excludes.end(),
                        [&](const std::string& e) { return contains_text(full, e); }))
            continue;
        selected.push_back(&t);
    }

    if (list_only) {
        for (const TestCase* t : selected) std::printf("%s\n", t->full_name().c_str());
        std::printf("%zu of %zu tests\n", selected.size(), registry().size());
        return 0;
    }
    if (selected.empty()) {
        std::printf("No tests match. Use --list to see all test names.\n");
        return 1;
    }

    std::printf("hnsw-lite comprehensive tests | kernel: %s | %zu of %zu tests selected\n\n",
                isa_name(active_isa()), selected.size(), registry().size());

    using Clock = std::chrono::steady_clock;
    const auto run_start = Clock::now();
    std::vector<std::string> failed;
    std::string current_group;
    std::size_t run = 0, skipped = 0;

    for (const TestCase* t : selected) {
        ++run;
        if (t->group != current_group) {
            current_group = t->group;
            std::printf("[%s]\n", current_group.c_str());
        }
        g_failures = 0;
        alloc_hook::disarm();
        std::string error, skip_reason;
        bool was_skipped = false;
        const auto start = Clock::now();
        try {
            t->fn();
        } catch (const RequireFailed&) {
            // already reported
        } catch (const TestSkipped& s) {
            was_skipped = true;
            skip_reason = s.reason;
        } catch (const std::exception& e) {
            error = std::string("unexpected exception: ") + e.what();
        } catch (...) {
            error = "unexpected non-standard exception";
        }
        alloc_hook::disarm();  // a test that exits early must not leave the hook armed
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const bool ok = g_failures == 0 && error.empty();
        if (was_skipped && ok) {
            ++skipped;
            std::printf("  SKIP  %-45s %8.1f ms  (%s)\n", t->name.c_str(), ms, skip_reason.c_str());
            continue;
        }
        std::printf("  %s  %-45s %8.1f ms\n", ok ? "PASS" : "FAIL", t->name.c_str(), ms);
        if (!error.empty()) std::printf("        %s\n", error.c_str());
        if (!ok) {
            failed.push_back(t->full_name());
            if (fail_fast) break;
        }
    }

    const double total = std::chrono::duration<double>(Clock::now() - run_start).count();
    std::printf("\n%zu passed, %zu failed, %zu skipped, %zu not run, %ld checks, %.2f s\n",
                run - failed.size() - skipped, failed.size(), skipped, selected.size() - run,
                g_checks.load(), total);
    if (!failed.empty()) {
        std::printf("Failed tests:\n");
        for (const auto& name : failed) std::printf("  %s\n", name.c_str());
        return 1;
    }
    std::printf("All selected tests passed.\n");
    return 0;
}