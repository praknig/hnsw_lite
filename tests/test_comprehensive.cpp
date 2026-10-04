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
 * Groups: layer1, layer2, helpers, flat, hnsw, robustness, concurrency, e2e
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
void check_graph(const HnswIndex& index, double min_reachable) {
    const Storage& st = index.storage();
    const GraphStorage& g = st.graph();
    const std::size_t n = st.size();
    if (n == 0) {
        CHECK(index.entry_point() == kEmpty && index.max_level() == -1);
        return;
    }
    REQUIRE(index.entry_point() < n);
    CHECK(g.level(index.entry_point()) == index.max_level());
    bool structure_ok = true;
    for (NodeId node = 0; node < n; ++node) {
        if (g.level(node) > index.max_level()) structure_ok = false;
        for (int level = 0; level <= g.level(node); ++level) {
            auto slots = g.links(node, level);
            const std::size_t count = GraphStorage::count(slots);
            if (count > (level == 0 ? g.M0() : g.M())) structure_ok = false;
            std::set<NodeId> seen;
            for (std::size_t i = 0; i < count; ++i) {
                const NodeId nb = slots[i];
                if (nb >= n || nb == node || g.level(nb) < level || !seen.insert(nb).second)
                    structure_ok = false;
            }
            for (std::size_t i = count; i < slots.size(); ++i)
                if (slots[i] != kEmpty) structure_ok = false;
        }
    }
    CHECK(structure_ok);

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
    std::size_t live = 0, live_reached = 0;
    for (NodeId node = 0; node < n; ++node) {
        if (st.ids().is_deleted(node)) continue;
        ++live;
        live_reached += reached[node];
    }
    if (live > 0) CHECK(double(live_reached) / double(live) >= min_reachable);
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

TEST(flat, removed_id_not_reusable) {
    FlatIndex f(2, Metric::L2);
    f.add(7, std::vector<float>{0, 0});
    f.remove(7);
    CHECK_THROWS_AS(std::invalid_argument, f.add(7, std::vector<float>{1, 1}));
    CHECK(f.size() == 0);
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
        CHECK(same_results(got, reference_search(data, removed, q, Metric::L2, 20, 1000)));
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

TEST(hnsw, removed_id_not_reusable) {
    HnswIndex h(2, Metric::L2);
    h.add(7, std::vector<float>{0, 0});
    h.remove(7);
    CHECK_THROWS_AS(std::invalid_argument, h.add(7, std::vector<float>{1, 1}));
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
    CHECK(h.storage().size() == 2);  // still stored, only flagged
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
    h.add(5000, data[0]);
    h.add(5001, data[1]);
    auto r = h.search(data[0], 5, 100);
    CHECK(r.size() == 2 && r[0].id == 5000);
    check_graph(h, 1.0);
}

TEST(hnsw, new_nodes_do_not_link_to_removed) {
    auto data = clustered(600, 16, 6, 74);
    HnswIndex h(16, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) h.add(i, data[i]);
    for (std::size_t i = 0; i < 300; i += 3) h.remove(i);
    for (std::size_t i = 300; i < 600; ++i) h.add(i, data[i]);
    const Storage& st = h.storage();
    bool ok = true;
    for (NodeId node = 300; node < 600; ++node)  // nodes inserted after the removals
        for (int l = 0; l <= st.graph().level(node); ++l)
            for (NodeId nb : st.graph().links(node, l)) {
                if (nb == kEmpty) break;
                ok = ok && !st.ids().is_deleted(nb);
            }
    CHECK(ok);
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
/// failure that reaches the caller, `verify` checks the object is still sound.
/// Stops when `op` completes without any injected failure firing. Returns the
/// number of allocation points tried, or -1 if `op` never completed.
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
        } else if (!alloc_hook::fired) {
            return n;  // completed with no failure injected: every point was tried
        }
        // A failure fired but was handled internally: keep going.
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
    std::uint64_t next_id = 10000;  // a failed insert may keep its ID taken, so use a new one each try
    std::uint64_t last_id = 0;
    const long points = sweep_allocation_failures(
        [&] {
            last_id = next_id++;
            h.add(last_id, data[300]);
        },
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
        "Groups: layer1, layer2, helpers, flat, hnsw, robustness, concurrency, e2e\n"
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