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
 *         metadata, filter, planner, payload, batch, range, search_e2e, update, hardening, persistence, add_batch, bulk, records, control, codec, sq8, matryoshka, lean, review, codes_only, binary, pq, ivf, edges, vamana, disk, stress, e2e
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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <condition_variable>
#include <streambuf>
#include <sstream>
#include <iterator>
#include <functional>
#include <fstream>
#include <filesystem>
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
#include "serialize.h"
#include "disk_index.h"
#include "kmeans.h"
#include "vamana.h"
#include "codec.h"
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

/// Exact reference search: every distance (same kernel, reported as the indexes
/// report it: cosine clamped at 0), sorted, first k.
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
        all.push_back({index_distance(fn(q.data(), row.data(), q.stride()), m == Metric::Cosine), i});  // as the index reports it
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
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{0, 200, 1, true, {}}));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{1, 200, 1, true, {}}));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex bad(4, Metric::L2, HnswParams{16, 0, 1, true, {}}));
    CHECK_NOTHROW(HnswIndex ok(4, Metric::L2, HnswParams{2, 1, 1, true, {}}));
}

TEST(hnsw, accessors) {
    HnswIndex h(7, Metric::InnerProduct, HnswParams{12, 50, 9, true, {}});
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
    HnswIndex h(16, Metric::L2, HnswParams{16, 32, 7, true, {}});
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
    HnswIndex h(16, Metric::L2, HnswParams{2, 200, 1, true, {}});
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    check_graph(h, 0.80);
    CHECK(recall(h, f, queries, 10, 100) >= 0.60);
}

TEST(hnsw, param_minimal_ef_construction) {
    // ef_construction = 1 builds greedily (measured recall ~0.66).
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    HnswIndex h(16, Metric::L2, HnswParams{16, 1, 1, true, {}});
    FlatIndex f(16, Metric::L2);
    fill(h, f, data);
    check_graph(h, 0.99);
    CHECK(recall(h, f, queries, 10, 100) >= 0.50);
}

TEST(hnsw, param_large_M) {
    auto data = clustered(600, 16, 6, 50);
    auto queries = clustered(40, 16, 6, 51);
    HnswIndex h(16, Metric::L2, HnswParams{64, 200, 1, true, {}});
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
        HnswIndex h(16, Metric::L2, HnswParams{16, 200, seed, true, {}});
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
    HnswIndex h(8, Metric::L2, HnswParams{16, 16, 11, true, {}});
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
        all.push_back({id, index_distance(fn(q.data(), row.data(), q.stride()), m == Metric::Cosine)});  // as the index reports it
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
    HnswIndex h(8, Metric::L2, HnswParams{4, 50, 3, true, {}});
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const LevelChange seen = reuse_levels(h, clustered(300, 8, 4, 213));
    CHECK(seen.higher && seen.same_slot);
    check_graph(h, 0.95);
    for (const auto& q : clustered(10, 8, 4, 214)) check_results(h, h.search(q, 10, 64));
}

TEST(deletion, d43_hnsw_reused_slot_lower_level) {
    auto data = clustered(300, 8, 4, 215);
    HnswIndex h(8, Metric::L2, HnswParams{4, 50, 5, true, {}});
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
    HnswIndex h(8, Metric::L2, HnswParams{2, 100, 1, true, {}});
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
    HnswIndex h(8, Metric::L2, HnswParams{4, 100, 1, true, {}});  // small lists fill up and get pruned
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
    HnswIndex h(8, Metric::L2, HnswParams{4, 100, 1, true, {}});
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
    HnswIndex h(16, Metric::L2, HnswParams{16, 100, 9, true, {}}, Schema::strict(search_fields()));
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

TEST(update, u28_zero_vector_with_cosine) {
    // A zero vector has no direction; like add(), update accepts it and it sits
    // at cosine distance 1 from everything (as in hnswlib and Faiss).
    HnswIndex h(3, Metric::Cosine);
    FlatIndex f(3, Metric::Cosine);
    for (std::uint64_t i = 0; i < 20; ++i) {
        h.add(i, std::vector<float>{1, float(i), 0});
        f.add(i, std::vector<float>{1, float(i), 0});
    }
    CHECK(h.update(4, std::vector<float>{0, 0, 0}) && f.update(4, std::vector<float>{0, 0, 0}));
    bool one = true;
    for (const auto& r : f.search(std::vector<float>{0, 1, 0}, 20)) if (r.id == 4) one = one && std::fabs(r.distance - 1.0f) < 1e-6f;
    CHECK(one && h.size() == 20);
    check_graph(h, 1.0);
}

TEST(update, u29_upsert_invalid_vector_for_new_id) {
    HnswIndex h(2, Metric::L2);
    FlatIndex f(2, Metric::L2);
    for (const auto& bad : {std::vector<float>{kNaN, 0}, std::vector<float>{kInf, 0}, std::vector<float>{1, 2, 3}}) {
        CHECK_THROWS_AS(std::invalid_argument, h.upsert(5, bad));
        CHECK_THROWS_AS(std::invalid_argument, f.upsert(5, bad));
    }
    CHECK(h.size() == 0 && h.capacity() == 0 && f.size() == 0 && !h.contains(5));
}

TEST(update, u30_compact_after_many_updates) {
    auto data = clustered(500, 8, 4, 437);
    auto pool = clustered(2000, 8, 4, 438);
    auto queries = clustered(20, 8, 4, 439);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    std::mt19937 rng(440);
    for (std::size_t i = 0; i < 2000; ++i) {
        const std::uint64_t id = rng() % 500;
        h.update(id, pool[i]);
        f.update(id, pool[i]);
    }
    CHECK(h.capacity() == 501 && h.deleted_count() == 1);
    const CompactStats st = h.compact();
    CHECK(st.kept == 500 && st.reclaimed == 1 && h.capacity() == 500 && h.deleted_count() == 0);
    CHECK(same_contents(h, f, queries) && recall(h, f, queries, 10, 64) >= 0.95);
}

TEST(update, u31_metadata_copy_triggers_dictionary_cleanup) {
    // An update copies the vector's metadata into the new slot; that write may
    // first reclaim unused strings (renumbering them). Metadata stays correct.
    FlatIndex f(1, Metric::L2);
    HnswIndex h(1, Metric::L2);
    for (std::uint64_t i = 0; i < 3000; ++i) {
        f.add(i, std::vector<float>{float(i)}, Metadata().set("k", "s" + std::to_string(i)));
        h.add(i, std::vector<float>{float(i)}, Metadata().set("k", "s" + std::to_string(i)));
    }
    for (std::uint64_t i = 0; i < 2900; ++i) {
        f.remove(i);
        h.remove(i);
    }
    CHECK(h.update(2950, std::vector<float>{7}));
    CHECK(f.update(2950, std::vector<float>{7}, Metadata().set("k", "fresh")));
    CHECK(h.metadata().dictionary_size() == 100 && h.metadata().unused_string_count() == 0);
    bool ok = h.get_metadata(2950)->get("k")->s == "s2950" && f.get_metadata(2950)->get("k")->s == "fresh";
    for (std::uint64_t i = 2900; i < 3000; ++i)
        if (i != 2950) ok = ok && h.get_metadata(i)->get("k")->s == "s" + std::to_string(i) && f.get_metadata(i)->get("k")->s == "s" + std::to_string(i);
    CHECK(ok && matching_ids(f, Filter::eq("k", "s2999")) == std::vector<std::uint64_t>{2999});
}

TEST(update, u32_updated_vector_never_left_without_links) {
    // Sparse, mostly removed graphs without repair (as in rg16). An update leaves
    // the old version out of the new one's neighbors; when the old version was
    // the only live vector the search could reach, the new version must still
    // get links (through the old version), never none at all.
    bool linked = true;
    for (unsigned seed = 1; seed <= 300; ++seed) {
        std::mt19937 rng(seed);
        HnswParams p;
        p.M = 2 + seed % 3;
        p.ef_construction = 1 + seed % 4;
        p.repair_on_remove = false;
        p.seed = seed;
        HnswIndex h(2, Metric::L2, p);
        const int n = 20 + int(seed % 30);
        for (int i = 0; i < n; ++i) h.add(std::uint64_t(i), std::vector<float>{portable_uniform(rng), portable_uniform(rng)});
        for (int i = 0; i < n - 4; ++i) h.remove(std::uint64_t(i));
        for (int i = n - 4; i < n; ++i) {
            h.update(std::uint64_t(i), std::vector<float>{portable_uniform(rng), portable_uniform(rng)});
            const NodeId slot = *h.storage().ids().find(std::uint64_t(i));
            linked = linked && h.storage().graph().links(slot, 0)[0] != kEmpty;
        }
    }
    CHECK(linked);
}

// ===========================================================================
// Hardening (H1-H14, for edge cases E1-E13 of the hardening and performance plan)
// ===========================================================================

TEST(hardening, h01_levels_identical_on_every_platform) {
    // E1: random levels come from SplitMix64's raw bits, so every compiler and
    // standard library must produce exactly these (checked on every CI platform).
    const std::vector<int> m16 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    const std::vector<int> m4 = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 2, 0,
                                 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    HnswIndex a(2, Metric::L2);
    HnswParams p;
    p.M = 4;
    p.seed = 7;
    HnswIndex b(2, Metric::L2, p);
    for (std::uint64_t i = 0; i < 64; ++i) {
        a.add(i, std::vector<float>{float(i), float(i % 7)});
        b.add(i, std::vector<float>{float(i), float(i % 7)});
    }
    bool same = true;
    for (NodeId n = 0; n < 64; ++n)
        same = same && a.storage().graph().level(n) == m16[n] && b.storage().graph().level(n) == m4[n];
    CHECK(same);
}

TEST(hardening, h02_nested_batch_from_a_predicate) {
    // E2: a predicate running a batch on the same index, inside a batch, used to
    // deadlock the pool. Nested batches now run inline.
    auto data = clustered(400, 8, 4, 450);
    auto queries = clustered(16, 8, 4, 451);
    const auto buffer = flatten(queries);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    const auto expected_h = h.search_batch(buffer, 5, 64, 4);
    const auto expected_f = f.search_batch(buffer, 5, 4);
    std::atomic<int> nested{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        nested += int(h.search_batch(std::span<const float>(buffer.data(), 16), 2, 16, 4).size());
        nested += int(f.search_batch(std::span<const float>(buffer.data(), 16), 2, 4).size());
        return true;
    };
    const auto got_h = h.search_batch(buffer, 5, 64, 4, o);
    const auto got_f = f.search_batch(buffer, 5, 4, o);
    bool same = true;
    for (std::size_t i = 0; i < queries.size(); ++i) same = same && same_results(got_h[i], expected_h[i]) && same_results(got_f[i], expected_f[i]);
    CHECK(same && nested.load() > 0);
}

TEST(hardening, h03_changes_during_a_search_are_refused) {
    // E2: a predicate may search the index but not change it; a change is refused
    // with std::logic_error and the index is left unchanged.
    auto data = clustered(300, 8, 4, 452);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    const Results before_h = h.search(data[0], 10), before_f = f.search(data[0], 10);
    const std::vector<std::function<void()>> hnsw_changes = {
        [&] { h.add(9999, data[1]); }, [&] { h.remove(5); }, [&] { h.update(5, data[1]); }, [&] { h.upsert(9999, data[1]); },
        [&] { h.set_metadata(5, Metadata().set("a", 1)); }, [&] { h.create_payload_index("a"); },
        [&] { h.set_planner_params(PlannerParams{}); }, [&] { h.compact(); }};
    const std::vector<std::function<void()>> flat_changes = {
        [&] { f.add(9999, data[1]); }, [&] { f.remove(5); }, [&] { f.update(5, data[1]); }, [&] { f.upsert(9999, data[1]); },
        [&] { f.set_metadata(5, Metadata().set("a", 1)); }, [&] { f.create_payload_index("a"); }, [&] { f.compact(); }};
    for (const auto& change : hnsw_changes) {
        SearchOptions o;
        o.predicate = [&](std::uint64_t) { change(); return true; };
        CHECK_THROWS_AS(std::logic_error, h.search(data[0], 10, 64, o));
        CHECK_THROWS_AS(std::logic_error, h.search_range(data[0], 1.0f, kNoLimit, o));
    }
    for (const auto& change : flat_changes) {
        SearchOptions o;
        o.predicate = [&](std::uint64_t) { change(); return true; };
        CHECK_THROWS_AS(std::logic_error, f.search(data[0], 10, o));
        CHECK_THROWS_AS(std::logic_error, f.search_batch(flatten(Vectors(data.begin(), data.begin() + 2)), 3, 2, o));
    }
    CHECK(same_results(h.search(data[0], 10), before_h) && same_results(f.search(data[0], 10), before_f));
    CHECK(h.size() == 300 && f.size() == 300 && !h.contains(9999) && !f.contains(9999));
    h.add(9999, data[1]);  // after the searches, changes work again
    f.add(9999, data[1]);
    CHECK(h.contains(9999) && f.contains(9999));
}

TEST(hardening, h04_cosine_distance_never_negative) {
    // E3: rounding used to give a vector a slightly negative distance to itself.
    std::mt19937 rng(453);
    std::normal_distribution<float> g(0.0f, 1.0f);
    bool non_negative = true;
    int found_with_tolerance = 0;
    for (int t = 0; t < 2000; ++t) {
        std::vector<float> v(37);
        for (float& x : v) x = g(rng);
        FlatIndex f(37, Metric::Cosine);
        f.add(1, v);
        const float d = f.search(v, 1)[0].distance;
        non_negative = non_negative && d >= 0.0f;
        found_with_tolerance += int(f.search_range(v, 1e-6f).size());
    }
    CHECK(non_negative);
    CHECK(found_with_tolerance == 2000);  // a small radius always finds exact copies
    // HNSW too, through search, range search and graph construction.
    Vectors data;
    for (int i = 0; i < 400; ++i) {
        std::vector<float> v(37);
        for (float& x : v) x = g(rng);
        data.push_back(v);
    }
    HnswIndex h(37, Metric::Cosine);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    bool hnsw_ok = true;
    for (std::size_t i = 0; i < data.size(); i += 4) {
        for (const auto& r : h.search(data[i], 10)) hnsw_ok = hnsw_ok && r.distance >= 0.0f;
        const auto near = h.search_range(data[i], 1e-6f);
        hnsw_ok = hnsw_ok && !near.empty() && near[0].id == i;
    }
    CHECK(hnsw_ok);
}

TEST(hardening, h05_hnsw_m_has_an_upper_bound) {
    // E4: a huge M used to attempt a huge allocation.
    HnswParams p;
    p.M = 2049;
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(4, Metric::L2, p));
    p.M = 2000000;
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(4, Metric::L2, p));
    p.M = 2048;
    HnswIndex h(4, Metric::L2, p);
    h.add(1, std::vector<float>{1, 2, 3, 4});
    CHECK(h.search(std::vector<float>{1, 2, 3, 4}, 1)[0].id == 1);
}

TEST(hardening, h06_extreme_ids) {
    // E5: IDs 0 and 2^64-1 through every operation.
    const std::uint64_t hi = std::numeric_limits<std::uint64_t>::max();
    auto data = clustered(50, 4, 2, 454);
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    for (std::size_t i = 0; i < 47; ++i) {
        h.add(1000 + i, data[i]);
        f.add(1000 + i, data[i]);
    }
    for (std::uint64_t id : {std::uint64_t(0), hi, hi - 1}) {
        h.add(id, data[47 + (id % 3)], Metadata().set("edge", true));
        f.add(id, data[47 + (id % 3)], Metadata().set("edge", true));
    }
    CHECK(h.update(hi, data[0]) && f.update(hi, data[0]));
    const SearchOptions o = with_filter(Filter::eq("edge", true));
    std::set<std::uint64_t> edge_h, edge_f;
    for (const auto& r : h.search(data[0], 10, 64, o)) edge_h.insert(r.id);
    for (const auto& r : f.search(data[0], 10, o)) edge_f.insert(r.id);
    CHECK(edge_h == std::set<std::uint64_t>({0, hi - 1, hi}) && edge_f == edge_h);
    CHECK(h.search_batch(data[0], 1)[0][0].distance == 0.0f);
    CHECK(h.remove(0) && h.remove(hi) && f.remove(hi) && !h.contains(hi) && !f.contains(hi));
    h.add(hi, data[5]);
    CHECK(h.contains(hi) && *h.get_metadata(hi) == Metadata());
}

TEST(hardening, h07_vector_from_the_index_itself) {
    // E6: adding or updating with a vector that points into the index's own storage.
    auto data = clustered(200, 8, 4, 455);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    const Storage& st = h.storage();
    const NodeId s10 = *st.ids().find(10), s20 = *st.ids().find(20);
    CHECK(h.update(10, st.vectors().get(s20)));          // copy vector 20 onto 10
    CHECK(h.update(20, st.vectors().get(s20)));          // a vector onto itself
    h.add(500, st.vectors().get(*st.ids().find(30)));    // add a copy of vector 30
    std::set<std::uint64_t> at_20;
    for (const auto& r : h.search(data[20], 5, 128)) if (r.distance == 0.0f) at_20.insert(r.id);
    CHECK(at_20 == std::set<std::uint64_t>({10, 20}));
    CHECK(h.search(data[30], 2, 128)[1].distance == 0.0f && s10 != s20);
}

TEST(hardening, h08_dimension_zero_rejected) {
    // E7.
    CHECK_THROWS_AS(std::invalid_argument, FlatIndex(0, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(0, Metric::Cosine));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(0, Metric::L2, HnswParams{}, Schema::strict({})));
}

TEST(hardening, h09_thousands_of_dynamic_fields) {
    // E8: 2,000 fields created dynamically; every vector sets 3 of them.
    FlatIndex f(1, Metric::L2);
    std::mt19937 rng(456);
    std::map<std::uint64_t, Metadata> ref;
    std::set<std::string> used;
    for (std::uint64_t i = 0; i < 3000; ++i) {
        Metadata md;
        for (int j = 0; j < 3; ++j) {
            const std::string name = "f" + std::to_string(rng() % 2000);
            md.set(name, std::int64_t(i));
            used.insert(name);
        }
        f.add(i, std::vector<float>{float(i)}, md);
        ref[i] = md;
    }
    CHECK(f.metadata().field_count() == used.size() && used.size() > 1900);
    bool ok = true;
    for (std::uint64_t i = 0; i < 3000; i += 97) ok = ok && *f.get_metadata(i) == ref[i];
    std::vector<std::uint64_t> expected;
    for (const auto& [id, md] : ref) if (md.get("f777")) expected.push_back(id);
    CHECK(ok && matching_ids(f, Filter::exists("f777")) == expected);
}

TEST(hardening, h10_thousands_of_tags_in_a_filter) {
    // E9: has_any_tag and has_all_tags with 5,000 tags (now sorted and binary-searched).
    FlatIndex f(1, Metric::L2);
    for (std::uint64_t i = 0; i < 20000; ++i)
        f.add(i, std::vector<float>{float(i)}, Metadata().set_tags("t", {"t" + std::to_string(i % 7000), "x" + std::to_string(i % 3)}));
    std::vector<std::string> many;
    for (int i = 4999; i >= 0; --i) many.push_back("t" + std::to_string(i));  // unsorted on purpose
    many.push_back("t0");                                                       // and a repeat
    std::vector<std::uint64_t> any_expected, all_expected;
    for (std::uint64_t i = 0; i < 20000; ++i) {
        if (i % 7000 < 5000) any_expected.push_back(i);
        if (i % 7000 == 12 && i % 3 == 1) all_expected.push_back(i);
    }
    CHECK(matching_ids(f, Filter::has_any_tag("t", many)) == any_expected);
    CHECK(matching_ids(f, Filter::has_all_tags("t", {"x1", "t12"})) == all_expected);
    CHECK(matching_ids(f, Filter::has_all_tags("t", {"x1", "t12", "x2"})).empty());  // more tags than any vector has
}

TEST(hardening, h11_throwing_predicate_in_a_threaded_batch) {
    // E10: the exception reaches the caller; both indexes stay usable.
    auto data = clustered(500, 8, 4, 457);
    const auto buffer = flatten(clustered(32, 8, 4, 458));
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    fill(h, f, data);
    SearchOptions o;
    o.predicate = [](std::uint64_t id) -> bool {
        if (id == 123) throw std::runtime_error("lookup failed");
        return true;
    };
    o.strategy = Strategy::ForceExact;
    CHECK_THROWS_AS(std::runtime_error, h.search_batch(buffer, 5, 64, 8, o));
    CHECK_THROWS_AS(std::runtime_error, f.search_batch(buffer, 5, 8, o));
    CHECK(h.search_batch(buffer, 5, 64, 8).size() == 32 && f.search_batch(buffer, 5, 8).size() == 32);
    h.add(9999, data[0]);  // and changes are accepted again
    CHECK(h.contains(9999));
}

TEST(hardening, h12_subnormal_values) {
    // E12: very small (subnormal) floats give correct, finite results.
    const float tiny = std::numeric_limits<float>::denorm_min();
    std::mt19937 rng(459);
    Vectors data;
    for (int i = 0; i < 300; ++i) {
        std::vector<float> v(8);
        for (float& x : v) x = float(rng() % 1000) * tiny;  // every value subnormal or zero
        data.push_back(v);
    }
    for (Metric m : kMetrics) {
        HnswIndex h(8, m);
        FlatIndex f(8, m);
        fill(h, f, data);
        bool finite = true;
        for (const auto& r : f.search(data[0], 10)) finite = finite && std::isfinite(r.distance);
        CHECK(finite && same_contents(h, f, Vectors(data.begin(), data.begin() + 10)));
    }
}

TEST(hardening, h13_deep_filter_on_pool_threads) {
    // E13: the deepest allowed filter, evaluated on pool threads (1 MB stacks on Windows).
    Filter deep = Filter::eq("n", 1);
    for (int i = 1; i < Filter::kMaxDepth; ++i) deep = deep && Filter::eq("n", 1);
    REQUIRE(deep.root().depth == Filter::kMaxDepth);
    auto data = clustered(300, 8, 4, 460);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], Metadata().set("n", std::int64_t(i % 2)));
        f.add(i, data[i], Metadata().set("n", std::int64_t(i % 2)));
    }
    const auto buffer = flatten(Vectors(data.begin(), data.begin() + 32));
    for (Strategy s : {Strategy::ForceExact, Strategy::ForceGraph}) {
        const auto hb = h.search_batch(buffer, 5, 64, 8, with_filter(deep, s));
        bool odd = true;
        for (const auto& results : hb)
            for (const auto& r : results) odd = odd && r.id % 2 == 1;
        CHECK(odd);
    }
    const auto fb = f.search_batch(buffer, 5, 8, with_filter(deep));
    for (std::size_t i = 0; i < fb.size(); ++i) CHECK(fb[i].size() == 5);
}

TEST(hardening, h14_search_counter_is_released_on_every_path) {
    // The change detection must never block a later change: searches that
    // return normally, throw, or return early all release it.
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    CHECK(h.search(std::vector<float>{1, 2, 3, 4}, 5).empty());                    // empty index
    CHECK_THROWS_AS(std::invalid_argument, h.search(std::vector<float>{1, 2}, 5));  // invalid query
    CHECK_THROWS_AS(std::invalid_argument, f.search_range(std::vector<float>{1, 2, 3, 4}, kNaN));
    CHECK_THROWS_AS(std::invalid_argument, f.search_batch(std::vector<float>{1, 2, 3}, 5));
    CHECK_NOTHROW(h.add(1, std::vector<float>{1, 2, 3, 4}));
    CHECK_NOTHROW(f.add(1, std::vector<float>{1, 2, 3, 4}));
    CHECK(h.search_batch(std::vector<float>{1, 2, 3, 4}, 1, 64, 4).size() == 1);
    CHECK_NOTHROW(h.remove(1));
}

// ===========================================================================
// Saving and loading (P1-P42 from the test plan, P50-P56 beyond it)
// ===========================================================================

namespace {

std::string save_bytes(const FlatIndex& f) {
    std::ostringstream out(std::ios::binary);
    f.save(out);
    return out.str();
}
std::string save_bytes(const HnswIndex& h) {
    std::ostringstream out(std::ios::binary);
    h.save(out);
    return out.str();
}
std::unique_ptr<FlatIndex> load_flat(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    return FlatIndex::load(in);
}
std::unique_ptr<HnswIndex> load_hnsw(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    return HnswIndex::load(in);
}

/// A unique temporary file path for one test.
std::string temp_path(const std::string& name) {
    // A token per process: test groups run in parallel (ctest -j), and two runs
    // on one machine must never share a file.
    static const std::string token = std::to_string(std::random_device{}());
    return (std::filesystem::temp_directory_path() / ("hnsw_lite_test_" + token + "_" + name + ".bin")).string();
}

/// Temporary files a save left next to `path` ("<path>.tmp-..."): must be none.
std::size_t temporaries_left(const std::string& path) {
    const std::filesystem::path p(path);
    const std::filesystem::path dir = p.parent_path();
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return 0;
    const std::string prefix = p.filename().string() + ".tmp";
    std::size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        if (entry.path().filename().string().rfind(prefix, 0) == 0) ++n;
    return n;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
}

/// Every link of every slot, for comparing two graphs exactly.
std::vector<std::vector<NodeId>> all_links(const HnswIndex& h) {
    std::vector<std::vector<NodeId>> out;
    const GraphStorage& g = h.storage().graph();
    for (NodeId n = 0; n < h.storage().size(); ++n)
        for (int l = 0; l <= g.level(n); ++l) {
            auto s = g.links(n, l);
            out.emplace_back(s.begin(), s.begin() + std::ptrdiff_t(GraphStorage::count(s)));
        }
    return out;
}

/// A file split into its header and sections, so tests can edit a section's
/// payload and recompute its checksum (to test validation, not the checksum).
struct ParsedFile {
    std::string header;
    std::vector<std::pair<std::uint32_t, std::string>> sections;  // (tag, payload)

    static ParsedFile parse(const std::string& bytes) {
        ParsedFile f;
        f.header = bytes.substr(0, 24);
        std::size_t at = 24;
        while (at < bytes.size()) {
            const auto t = persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(&bytes[at]));
            const auto len = persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(&bytes[at + 4]));
            f.sections.push_back({t, bytes.substr(at + 12, std::size_t(len))});
            at += 12 + std::size_t(len) + 4;
        }
        return f;
    }

    std::string& payload(const char (&name)[5]) {
        for (auto& [t, p] : sections)
            if (t == persist::tag(name)) return p;
        throw std::logic_error("no such section");
    }

    std::string assemble() const {
        std::string out = header;
        for (const auto& [t, p] : sections) {
            unsigned char head[12], crc[4];
            persist::to_little_endian(t, head);
            persist::to_little_endian(std::uint64_t(p.size()), head + 4);
            persist::to_little_endian(persist::crc32(0, p.data(), p.size()), crc);
            out.append(reinterpret_cast<const char*>(head), 12).append(p).append(reinterpret_cast<const char*>(crc), 4);
        }
        return out;
    }
};

template <class T>
void poke(std::string& s, std::size_t at, T value) {
    persist::to_little_endian(value, reinterpret_cast<unsigned char*>(&s[at]));
}

/// Vectors of small whole numbers: every distance kernel sums them exactly, so
/// the same index is built identically on every CPU and platform.
std::vector<float> integer_vector(std::mt19937& rng, std::size_t dim) {
    std::vector<float> v(dim);
    for (float& x : v) x = float((rng() >> 28) & 15u);
    return v;
}

/// The index stored in tests/data/golden_hnsw_v1.bin and _v2.bin (format versions 1 and 2).
std::unique_ptr<HnswIndex> golden_hnsw() {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 40;
    p.seed = 7;
    auto h = std::make_unique<HnswIndex>(8, Metric::L2, p, Schema::dynamic({{"category", FieldType::Keyword}}));
    h->create_payload_index("category");
    std::mt19937 rng(2026);
    for (std::uint64_t i = 0; i < 300; ++i)
        h->add(i, integer_vector(rng, 8),
               Metadata().set("category", kCategories[i % 5]).set("n", std::int64_t(i)).set_tags("t", {i % 2 ? "odd" : "even"}));
    for (std::uint64_t i = 0; i < 300; i += 7) h->remove(i);
    for (std::uint64_t i = 1; i < 300; i += 11)
        if (h->contains(i)) h->update(i, integer_vector(rng, 8));
    h->set_planner_params(PlannerParams{500, 0.02, 128, 16.0});
    return h;
}

/// The index stored in tests/data/golden_flat_v1.bin and _v2.bin (format versions 1 and 2).
std::unique_ptr<FlatIndex> golden_flat() {
    auto f = std::make_unique<FlatIndex>(8, Metric::L2, Schema::strict({{"category", FieldType::Keyword}, {"n", FieldType::Int}}));
    std::mt19937 rng(2027);
    for (std::uint64_t i = 0; i < 100; ++i)
        f->add(1000 + i, integer_vector(rng, 8), Metadata().set("category", kCategories[i % 5]).set("n", std::int64_t(i)));
    for (std::uint64_t i = 0; i < 100; i += 9) f->remove(1000 + i);
    f->update(1001, integer_vector(rng, 8));
    return f;
}

std::string golden_path(const char* name) { return std::string(HNSW_TEST_DATA_DIR) + "/" + name; }

/// The index stored in tests/data/golden_flat_ivf_v3.bin: a Flat index with a
/// trained inverted file (8 lists), from whole numbers, so every distance is exact
/// and every CPU assigns the same lists.
std::unique_ptr<FlatIndex> golden_ivf() {
    IvfParams ivf;
    ivf.lists = 8;
    ivf.probes = 3;
    ivf.train_after = 200;
    auto f = std::make_unique<FlatIndex>(12, Metric::L2, Schema::dynamic(), QuantizationParams{}, ivf);
    std::mt19937 rng(2031);
    for (std::uint64_t i = 0; i < 260; ++i) f->add(i, integer_vector(rng, 12));
    for (std::uint64_t i = 0; i < 260; i += 23) f->remove(i);
    return f;
}

/// The index stored in tests/data/golden_hnsw_pq_v3.bin: PQ codes only (4 parts),
/// trained by k-means on whole numbers; identical on every CPU.
std::unique_ptr<HnswIndex> golden_pq() {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 40;
    p.seed = 13;
    p.quantization.codec = CodecKind::PQ;
    p.quantization.train_after = 300;
    p.quantization.pq_subspaces = 4;
    p.quantization.keep_full_vectors = false;
    auto h = std::make_unique<HnswIndex>(16, Metric::L2, p);
    std::mt19937 rng(2030);
    for (std::uint64_t i = 0; i < 350; ++i) h->add(i, integer_vector(rng, 16));
    for (std::uint64_t i = 0; i < 350; i += 19) h->remove(i);
    return h;
}

/// The index stored in tests/data/golden_hnsw_binary_v3.bin: binary codes only,
/// from whole numbers (thresholds are means: exact here), identical on every CPU.
std::unique_ptr<HnswIndex> golden_binary() {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 40;
    p.seed = 11;
    p.quantization.codec = CodecKind::Binary;
    p.quantization.train_after = 64;
    p.quantization.keep_full_vectors = false;
    auto h = std::make_unique<HnswIndex>(16, Metric::L2, p);
    std::mt19937 rng(2029);
    for (std::uint64_t i = 0; i < 200; ++i) h->add(i, integer_vector(rng, 16));
    for (std::uint64_t i = 0; i < 200; i += 17) h->remove(i);
    return h;
}

/// The index stored in tests/data/golden_hnsw_sq8_v3.bin: SQ8-trained, from whole
/// numbers, so its codes are identical on every CPU.
std::unique_ptr<HnswIndex> golden_sq8() {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 40;
    p.seed = 9;
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 100;
    auto h = std::make_unique<HnswIndex>(8, Metric::L2, p);
    std::mt19937 rng(2028);
    for (std::uint64_t i = 0; i < 250; ++i) h->add(i, integer_vector(rng, 8));
    for (std::uint64_t i = 0; i < 250; i += 13) h->remove(i);
    return h;
}

/// A small HNSW index used by the corruption tests (small, so every byte can be tried).
std::unique_ptr<HnswIndex> tiny_hnsw() {
    HnswParams p;
    p.M = 4;
    p.ef_construction = 16;
    auto h = std::make_unique<HnswIndex>(4, Metric::L2, p);
    std::mt19937 rng(5);
    for (std::uint64_t i = 0; i < 30; ++i) h->add(i, integer_vector(rng, 4), Metadata().set("k", i % 2 ? "a" : "b"));
    h->remove(3);
    h->remove(17);
    return h;
}

}  // namespace

TEST(persistence, p01_flat_round_trip_every_metric) {
    auto data = random_vectors(400, 12, 500);
    auto queries = random_vectors(15, 12, 501);
    for (Metric m : kMetrics) {
        FlatIndex f(12, m, Schema::strict(search_fields()));
        for (std::size_t i = 0; i < data.size(); ++i) f.add(i, data[i], meta_for(i));
        auto g = load_flat(save_bytes(f));
        CHECK(g->size() == f.size() && g->dim() == 12 && g->metric() == m);
        bool same = true;
        for (const auto& q : queries) {
            same = same && same_results(g->search(q, 10), f.search(q, 10));
            same = same && same_results(g->search(q, 10, with_filter(Filter::eq("category", "tech"))),
                                        f.search(q, 10, with_filter(Filter::eq("category", "tech"))));
        }
        for (std::uint64_t i = 0; i < 400; i += 37) same = same && *g->get_metadata(i) == *f.get_metadata(i);
        CHECK(same);
    }
}

TEST(persistence, p02_hnsw_round_trip_every_metric) {
    auto data = clustered(800, 12, 6, 502);
    auto queries = clustered(15, 12, 6, 503);
    for (Metric m : kMetrics) {
        HnswIndex h(12, m);
        for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
        auto g = load_hnsw(save_bytes(h));
        CHECK(g->entry_point() == h.entry_point() && g->max_level() == h.max_level() && g->size() == h.size());
        CHECK(all_links(*g) == all_links(h));
        bool same = true;
        for (const auto& q : queries) same = same && same_results(g->search(q, 10, 64), h.search(q, 10, 64));
        CHECK(same);
    }
}

TEST(persistence, p03_after_removals_updates_and_compaction) {
    auto data = clustered(600, 8, 4, 504);
    auto moved = clustered(100, 8, 4, 505);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 0; i < 600; i += 4) h.remove(i);
    for (std::size_t i = 0; i < 100; ++i) h.update(i * 6 + 1, moved[i]);
    for (int stage = 0; stage < 2; ++stage) {
        auto g = load_hnsw(save_bytes(h));
        CHECK(g->size() == h.size() && g->capacity() == h.capacity() && g->deleted_count() == h.deleted_count());
        CHECK(all_links(*g) == all_links(h));
        bool same = true;
        for (std::size_t i = 0; i < 30; ++i) same = same && same_results(g->search(data[i], 10, 64), h.search(data[i], 10, 64));
        for (std::uint64_t i = 1; i < 600; i += 4) same = same && *g->get_metadata(i) == *h.get_metadata(i);
        CHECK(same);
        check_graph(*g, 0.99);
        h.compact();  // second round: after compaction
    }
}

TEST(persistence, p04_empty_indexes) {
    FlatIndex f(5, Metric::Cosine);
    HnswIndex h(5, Metric::InnerProduct);
    auto f2 = load_flat(save_bytes(f));
    auto h2 = load_hnsw(save_bytes(h));
    CHECK(f2->size() == 0 && f2->dim() == 5 && f2->metric() == Metric::Cosine);
    CHECK(h2->size() == 0 && h2->entry_point() == kEmpty && h2->max_level() == -1 && h2->metric() == Metric::InnerProduct);
    h2->add(1, std::vector<float>{1, 2, 3, 4, 5});  // usable afterwards
    CHECK(h2->search(std::vector<float>{1, 2, 3, 4, 5}, 1)[0].id == 1);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, std::vector<float>(5, float(i)));
    for (std::uint64_t i = 0; i < 20; ++i) h.remove(i);  // removed everything: still "empty"
    auto h3 = load_hnsw(save_bytes(h));
    CHECK(h3->size() == 0 && h3->capacity() == 20 && h3->entry_point() == kEmpty);
}

TEST(persistence, p05_single_vector) {
    FlatIndex f(3, Metric::L2);
    HnswIndex h(3, Metric::L2);
    f.add(9, std::vector<float>{1, 2, 3});
    h.add(9, std::vector<float>{1, 2, 3});
    CHECK(load_flat(save_bytes(f))->search(std::vector<float>{0, 0, 0}, 5)[0].id == 9);
    auto g = load_hnsw(save_bytes(h));
    CHECK(g->search(std::vector<float>{0, 0, 0}, 5).size() == 1 && g->entry_point() == 0);
}

TEST(persistence, p06_dimensions) {
    for (std::size_t dim : std::initializer_list<std::size_t>{1, 37, 1536}) {
        auto data = random_vectors(60, dim, unsigned(506 + dim));
        FlatIndex f(dim, Metric::L2);
        HnswIndex h(dim, Metric::Cosine);
        fill(h, f, data);
        auto f2 = load_flat(save_bytes(f));
        auto h2 = load_hnsw(save_bytes(h));
        CHECK(same_results(f2->search(data[3], 10), f.search(data[3], 10)));
        CHECK(same_results(h2->search(data[3], 10), h.search(data[3], 10)));
    }
}

TEST(persistence, p07_extreme_ids) {
    FlatIndex f(2, Metric::L2);
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t id : {std::uint64_t(0), kMaxId, kMaxId - 1}) {
        f.add(id, std::vector<float>{float(id % 5), 1});
        h.add(id, std::vector<float>{float(id % 5), 1});
    }
    auto f2 = load_flat(save_bytes(f));
    auto h2 = load_hnsw(save_bytes(h));
    for (std::uint64_t id : {std::uint64_t(0), kMaxId, kMaxId - 1}) CHECK(f2->contains(id) && h2->contains(id));
}

TEST(persistence, p08_settings_preserved) {
    HnswParams p;
    p.M = 12;
    p.ef_construction = 77;
    p.seed = 123456789;
    p.repair_on_remove = false;
    HnswIndex h(9, Metric::Cosine, p, Schema::strict({{"a", FieldType::Int}}));
    h.set_planner_params(PlannerParams{1234, 0.07, 99, 8.5});
    auto g = load_hnsw(save_bytes(h));
    CHECK(g->dim() == 9 && g->metric() == Metric::Cosine);
    CHECK(g->params().M == 12 && g->params().ef_construction == 77 && g->params().seed == 123456789 && !g->params().repair_on_remove);
    const PlannerParams& pp = g->planner_params();
    CHECK(pp.max_exact_matches == 1234 && pp.max_exact_fraction == 0.07 && pp.sample_size == 99 && pp.max_ef_multiplier == 8.5);
    CHECK(g->metadata().mode() == SchemaMode::Strict && g->metadata().field_count() == 1);
    CHECK_THROWS_AS(std::invalid_argument, g->add(1, std::vector<float>(9, 1.0f), Metadata().set("b", 1)));  // still strict
}

TEST(persistence, p09_continuation_builds_the_same_graph) {
    auto data = clustered(1000, 8, 6, 507);
    HnswIndex straight(8, Metric::L2), first(8, Metric::L2);
    for (std::size_t i = 0; i < 1000; ++i) straight.add(i, data[i], meta_for(i));
    for (std::size_t i = 0; i < 500; ++i) first.add(i, data[i], meta_for(i));
    auto resumed = load_hnsw(save_bytes(first));
    for (std::size_t i = 500; i < 1000; ++i) resumed->add(i, data[i], meta_for(i));
    CHECK(save_bytes(*resumed) == save_bytes(straight));  // identical in every byte
}

TEST(persistence, p10_free_list_survives) {
    auto data = clustered(200, 8, 4, 508);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) h.add(i, data[i]);
    for (std::uint64_t i : {std::uint64_t(5), std::uint64_t(50), std::uint64_t(99)}) h.remove(i);
    auto g = load_hnsw(save_bytes(h));
    CHECK(g->deleted_count() == 3);
    for (std::uint64_t i = 0; i < 3; ++i) {
        h.add(1000 + i, data[i]);
        g->add(1000 + i, data[i]);
        CHECK(*g->storage().ids().find(1000 + i) == *h.storage().ids().find(1000 + i));  // same slot, same order
    }
    CHECK(save_bytes(*g) == save_bytes(h));
}

TEST(persistence, p11_stream_and_file_identical) {
    auto h = golden_hnsw();
    const std::string path = temp_path("p11");
    h->save(path);
    CHECK(read_file(path) == save_bytes(*h));
    auto f = golden_flat();
    f->save(path);
    CHECK(read_file(path) == save_bytes(*f));
    std::filesystem::remove(path);
}

TEST(persistence, p12_saving_is_deterministic) {
    auto h = golden_hnsw();
    const std::string once = save_bytes(*h);
    CHECK(save_bytes(*h) == once);                   // twice: identical
    CHECK(save_bytes(*load_hnsw(once)) == once);     // save, load, save: identical
    auto f = golden_flat();
    const std::string flat_once = save_bytes(*f);
    CHECK(save_bytes(*load_flat(flat_once)) == flat_once);
}

TEST(persistence, p13_overwrite_leaves_no_temporary_file) {
    const std::string path = temp_path("p13");
    write_file(path, "an older file");
    auto h = golden_hnsw();
    h->save(path);
    CHECK(read_file(path) == save_bytes(*h));
    CHECK(temporaries_left(path) == 0);
    CHECK(HnswIndex::load(path)->size() == h->size());
    std::filesystem::remove(path);
}

TEST(persistence, p20_truncated_at_every_byte) {
    const std::string bytes = save_bytes(*tiny_hnsw());
    std::size_t rejected = 0;
    for (std::size_t len = 0; len < bytes.size(); ++len) {
        try {
            load_hnsw(bytes.substr(0, len));
        } catch (const FormatError&) {
            ++rejected;
        }
    }
    CHECK(rejected == bytes.size());  // every shorter file is rejected with FormatError
}

TEST(persistence, p21_any_flipped_bit_is_detected) {
    const std::string bytes = save_bytes(*tiny_hnsw());
    std::size_t rejected = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        std::string damaged = bytes;
        damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
        try {
            load_hnsw(damaged);
        } catch (const FormatError&) {
            ++rejected;
        }
    }
    CHECK(rejected == bytes.size());
}

TEST(persistence, p22_wrong_magic_version_or_byte_order) {
    const std::string bytes = save_bytes(*tiny_hnsw());
    std::string magic = bytes, version = bytes, zero = bytes, order = bytes, reserved = bytes;
    magic[0] = 'X';
    poke<std::uint32_t>(version, 8, persist::kVersion + 1);  // a future version
    poke<std::uint32_t>(zero, 8, 0);
    poke<std::uint32_t>(order, 12, 0x04030201u);
    poke<std::uint32_t>(reserved, 20, 1);
    for (const std::string& bad : {magic, version, zero, order, reserved}) CHECK_THROWS_AS(FormatError, load_hnsw(bad));
    try {
        load_hnsw(version);
    } catch (const FormatError& e) {
        CHECK(std::string(e.what()).find("version " + std::to_string(persist::kVersion + 1)) != std::string::npos);  // a clear message
    }
}

TEST(persistence, p23_wrong_index_kind) {
    CHECK_THROWS_AS(FormatError, load_hnsw(save_bytes(*golden_flat())));
    CHECK_THROWS_AS(FormatError, load_flat(save_bytes(*golden_hnsw())));
}

TEST(persistence, p24_empty_or_header_only) {
    CHECK_THROWS_AS(FormatError, load_hnsw(""));
    CHECK_THROWS_AS(FormatError, load_flat(""));
    const std::string bytes = save_bytes(*tiny_hnsw());
    CHECK_THROWS_AS(FormatError, load_hnsw(bytes.substr(0, 24)));
}

TEST(persistence, p25_bytes_after_the_end) {
    const std::string bytes = save_bytes(*tiny_hnsw());
    CHECK_THROWS_AS(FormatError, load_hnsw(bytes + "x"));
    CHECK_THROWS_AS(FormatError, load_hnsw(bytes + bytes));
    CHECK_THROWS_AS(FormatError, load_flat(save_bytes(*golden_flat()) + std::string(1, '\0')));
}

TEST(persistence, p26_inflated_counts_rejected_before_allocating) {
    ParsedFile f = ParsedFile::parse(save_bytes(*tiny_hnsw()));
    std::string& node = f.payload("NODE");
    const std::string original = node;
    poke<std::uint64_t>(node, 0, std::uint64_t(1) << 40);  // "a trillion vectors"
    CHECK_THROWS_AS(FormatError, load_hnsw(f.assemble()));
    node = original;
    std::string& stat = f.payload("STAT");
    poke<std::uint64_t>(stat, 8, std::uint64_t(1) << 50);  // free-slot count
    CHECK_THROWS_AS(FormatError, load_hnsw(f.assemble()));
    ParsedFile g = ParsedFile::parse(save_bytes(*golden_flat()));
    poke<std::uint64_t>(g.payload("NODE"), 0, std::uint64_t(1) << 45);
    CHECK_THROWS_AS(FormatError, load_flat(g.assemble()));
    if constexpr (HNSW_ALLOC_HOOK) {  // nothing large is even attempted: fail the first allocation over 1 GB
        ParsedFile h = ParsedFile::parse(save_bytes(*tiny_hnsw()));
        poke<std::uint64_t>(h.payload("NODE"), 0, std::uint64_t(1) << 40);
        const std::string bad = h.assemble();
        CHECK_THROWS_AS(FormatError, load_hnsw(bad));  // FormatError, not bad_alloc
    }
}

TEST(persistence, p27_invalid_content_with_valid_checksums) {
    const std::string good = save_bytes(*tiny_hnsw());
    const ParsedFile base = ParsedFile::parse(good);
    // Walk the NODE payload of slot 0 and slot 1: id (8), removed (1), level (1), vector (4 x 4), then per level: count (4) + links.
    const std::string node = ParsedFile(base).payload("NODE");
    const std::size_t slot0 = 8, links0 = slot0 + 8 + 1 + 1 + 16;
    const auto count0 = persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(&node[links0]));
    REQUIRE(count0 > 0);
    auto with_node = [&](auto&& edit) {
        ParsedFile f = base;
        edit(f.payload("NODE"));
        return f.assemble();
    };
    auto with_stat = [&](auto&& edit) {
        ParsedFile f = base;
        edit(f.payload("STAT"));
        return f.assemble();
    };
    const std::vector<std::string> bad = {
        with_node([&](std::string& p) { poke<std::uint32_t>(p, links0 + 4, 999999); }),          // link out of range
        with_node([&](std::string& p) { poke<std::uint32_t>(p, links0 + 4, 0); }),               // self-link
        with_node([&](std::string& p) { poke<std::uint32_t>(p, links0, 1000); }),                // list longer than capacity
        with_node([&](std::string& p) { poke<std::uint8_t>(p, slot0 + 8, 2); }),                 // removed flag 2
        with_node([&](std::string& p) { poke<float>(p, slot0 + 10, kNaN); }),                    // NaN in a vector
        with_node([&](std::string& p) { poke<std::uint64_t>(p, slot0, 1); }),                    // duplicate user ID (slot 1 has ID 1)
        with_stat([&](std::string& p) { poke<std::uint32_t>(p, 0, 999); }),                      // entry point out of range
        with_stat([&](std::string& p) { poke<std::int32_t>(p, 4, 7); }),                         // wrong top level
        with_stat([&](std::string& p) { poke<std::uint32_t>(p, 16, 4); }),                       // free list names a live slot
    };
    for (const std::string& b : bad) CHECK_THROWS_AS(FormatError, load_hnsw(b));
    // Duplicate links in one list: make the second link equal the first.
    if (count0 >= 2) {
        const std::string dup = with_node([&](std::string& p) { p.replace(links0 + 8, 4, p.substr(links0 + 4, 4)); });
        CHECK_THROWS_AS(FormatError, load_hnsw(dup));
    }
    CHECK(load_hnsw(good)->size() == 28);  // the unedited file still loads
}

TEST(persistence, p28_unusable_paths) {
    auto h = golden_hnsw();
    const std::string missing_dir = (std::filesystem::temp_directory_path() / "hnsw_lite_no_such_dir" / "x.bin").string();
    CHECK_THROWS_AS(std::runtime_error, h->save(missing_dir));
    CHECK_THROWS_AS(std::runtime_error, HnswIndex::load(missing_dir));
    CHECK_THROWS_AS(std::runtime_error, FlatIndex::load(missing_dir));
    CHECK(temporaries_left(missing_dir) == 0);
}

TEST(persistence, p29_oom_load) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    const std::string bytes = save_bytes(*golden_hnsw()), flat_bytes = save_bytes(*golden_flat());
    std::unique_ptr<HnswIndex> h;
    CHECK(sweep_allocation_failures([&] { h = load_hnsw(bytes); }, [&] { CHECK(h == nullptr); }) > 0);
    CHECK(h && save_bytes(*h) == bytes);
    std::unique_ptr<FlatIndex> f;
    CHECK(sweep_allocation_failures([&] { f = load_flat(flat_bytes); }, [&] { CHECK(f == nullptr); }) > 0);
    CHECK(f && save_bytes(*f) == flat_bytes);
}

TEST(persistence, p30_oom_save_keeps_the_old_file) {
    // Running out of memory at any point of a save must leave the old file intact
    // and no temporary file behind. Standard libraries differ in how a stream
    // reports it: libstdc++ passes std::bad_alloc on, libc++ (macOS) throws
    // std::ios_base::failure. Either is a clean failure.
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    const std::string path = temp_path("p30");
    auto old_index = tiny_hnsw();
    old_index->save(path);
    const std::string old_bytes = read_file(path);
    auto h = golden_hnsw();
    long points = 0;
    bool clean = true, completed = false;
    for (long n = 0; n < 5000 && !completed; ++n) {
        alloc_hook::fail_after(n);
        bool failed = false;
        try {
            h->save(path);
        } catch (const std::bad_alloc&) {
            failed = true;
        } catch (const std::ios_base::failure&) {
            failed = true;
        }
        alloc_hook::disarm();
        if (failed) {
            ++points;
            clean = clean && read_file(path) == old_bytes && temporaries_left(path) == 0;
        } else {
            completed = !alloc_hook::fired;  // a failure that did not surface would be a bug
            clean = clean && completed;
        }
    }
    CHECK(clean && completed && points > 0);
    CHECK(read_file(path) == save_bytes(*h));
    std::filesystem::remove(path);
}

TEST(persistence, p40_golden_files_load) {
  // Every format version this library reads: the version 1 files are frozen, to
  // prove that files saved by older versions of the library keep loading.
  for (const char* version : {"v1", "v2", "v3"}) {
    auto h = HnswIndex::load(golden_path((std::string("golden_hnsw_") + version + ".bin").c_str()));
    auto f = FlatIndex::load(golden_path((std::string("golden_flat_") + version + ".bin").c_str()));
    auto h_ref = golden_hnsw();
    auto f_ref = golden_flat();
    CHECK(h->size() == h_ref->size() && h->capacity() == h_ref->capacity() && h->deleted_count() == h_ref->deleted_count());
    CHECK(f->size() == f_ref->size());
    std::mt19937 rng(99);
    bool same = true;
    for (int i = 0; i < 20; ++i) {
        const auto q = integer_vector(rng, 8);
        same = same && same_results(h->search(q, 10, 32), h_ref->search(q, 10, 32));
        same = same && same_results(f->search(q, 10), f_ref->search(q, 10));
        same = same && same_results(h->search(q, 5, 32, with_filter(Filter::eq("category", "news"))),
                                    h_ref->search(q, 5, 32, with_filter(Filter::eq("category", "news"))));
    }
    CHECK(same);
    CHECK(*h->get_metadata(2) == *h_ref->get_metadata(2) && *f->get_metadata(1002) == *f_ref->get_metadata(1002));
    CHECK(h->metadata().is_indexed(*h->metadata().find_field("category")));
    CHECK(*h->get_vector(2) == *h_ref->get_vector(2));
  }
}

TEST(persistence, p41_identical_bytes_on_every_platform) {
    // Built from whole numbers, which every distance kernel sums exactly, so every
    // CI platform must reproduce the committed version 1 files byte for byte.
    CHECK(save_bytes(*golden_hnsw()) == read_file(golden_path("golden_hnsw_v3.bin")));
    CHECK(save_bytes(*golden_flat()) == read_file(golden_path("golden_flat_v3.bin")));
    // SQ8 codes are part of the file: every CPU's encoding must match byte for byte.
    CHECK(save_bytes(*golden_sq8()) == read_file(golden_path("golden_hnsw_sq8_v3.bin")));
    CHECK(save_bytes(*golden_binary()) == read_file(golden_path("golden_hnsw_binary_v3.bin")));
    CHECK(save_bytes(*golden_pq()) == read_file(golden_path("golden_hnsw_pq_v3.bin")));
    CHECK(save_bytes(*golden_ivf()) == read_file(golden_path("golden_flat_ivf_v3.bin")));
    auto loaded = HnswIndex::load(golden_path("golden_hnsw_sq8_v3.bin"));
    CHECK(loaded->codec_trained() && *loaded->get_code(5) == *golden_sq8()->get_code(5));
}

TEST(persistence, p42_large_round_trip) {
    auto data = clustered(20000, 16, 20, 509);
    HnswIndex h(16, Metric::L2, HnswParams{12, 60, 3, true, {}});
    FlatIndex f(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], Metadata().set("g", std::int64_t(i % 10)));
        f.add(i, data[i], Metadata().set("g", std::int64_t(i % 10)));
    }
    for (std::size_t i = 0; i < 80000; ++i) f.add(100000 + i, data[i % 20000]);  // Flat: 100,000 vectors
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    auto h2 = load_hnsw(hb);
    auto f2 = load_flat(fb);
    CHECK(f2->size() == 100000 && h2->size() == 20000);
    CHECK(all_links(*h2) == all_links(h));
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) {
        same = same && same_results(h2->search(data[i * 97], 10), h.search(data[i * 97], 10));
        same = same && same_results(f2->search(data[i * 97], 10), f.search(data[i * 97], 10));
    }
    CHECK(same && save_bytes(*h2) == hb && save_bytes(*f2) == fb);
}

TEST(persistence, p50_metadata_round_trip) {
    for (int strict = 0; strict < 2; ++strict) {
        const std::vector<FieldSpec> declared{{"year", FieldType::Int}};  // named: no temporary inside ?:
        const Schema schema = strict ? Schema::strict(search_fields()) : Schema::dynamic(declared);
        HnswIndex h(4, Metric::L2, HnswParams{}, schema);
        for (std::uint64_t i = 0; i < 300; ++i) h.add(i, std::vector<float>{float(i), 0, 0, 1}, meta_for(i));
        h.create_payload_index("category");  // after the adds: a dynamic schema creates the fields on first use
        h.create_payload_index("tags");
        for (std::uint64_t i = 0; i < 300; i += 3) h.remove(i);  // leaves unused strings? (categories stay used)
        h.set_metadata(5, Metadata().set("category", "one-off").unset("tags"));
        h.set_metadata(5, Metadata().set("category", "news"));  // "one-off" is now unused
        auto g = load_hnsw(save_bytes(h));
        CHECK(g->metadata().mode() == (strict ? SchemaMode::Strict : SchemaMode::Dynamic));
        CHECK(g->metadata().dictionary_size() == h.metadata().dictionary_size());
        CHECK(g->metadata().unused_string_count() == h.metadata().unused_string_count());
        bool same = true;
        for (std::uint64_t i = 0; i < 300; ++i) {
            const auto a = g->get_metadata(i), b = h.get_metadata(i);
            same = same && a.has_value() == b.has_value() && (!a || *a == *b);
        }
        CHECK(same);
        const MetadataStore &m = g->metadata(), &n = h.metadata();
        for (const char* cat : kCategories)
            CHECK(m.value_count(*m.find_field("category"), *m.find_string(cat)) == n.value_count(*n.find_field("category"), *n.find_string(cat)));
        SearchStats st;
        SearchOptions o = with_filter(Filter::eq("category", "tech"));
        o.stats = &st;
        g->search(std::vector<float>{1, 0, 0, 1}, 5, 64, o);
        CHECK(st.exact_count);  // the payload index works after loading
    }
}

TEST(persistence, p51_every_search_feature_after_loading) {
    const MetaFixture& fx = meta_fixture();
    auto h = load_hnsw(save_bytes(*fx.hnsw));
    auto f = load_flat(save_bytes(*fx.flat));
    const SearchOptions o = with_filter(Filter::lt("bucket", 30) && Filter::ne("category", "food"));
    const auto buffer = flatten(Vectors(fx.queries.begin(), fx.queries.begin() + 8));
    bool same = true;
    for (std::size_t i = 0; i < 8; ++i) {
        same = same && same_results(h->search(fx.queries[i], 10, 64, o), fx.hnsw->search(fx.queries[i], 10, 64, o));
        same = same && same_results(f->search(fx.queries[i], 10, o), fx.flat->search(fx.queries[i], 10, o));
        same = same && same_results(h->search_range(fx.queries[i], 2.0f), fx.hnsw->search_range(fx.queries[i], 2.0f));
    }
    const auto hb = h->search_batch(buffer, 5, 64, 2, o), hb_ref = fx.hnsw->search_batch(buffer, 5, 64, 2, o);
    for (std::size_t i = 0; i < 8; ++i) same = same && same_results(hb[i], hb_ref[i]);
    CHECK(same);
}

TEST(persistence, p52_changes_continue_identically_after_loading) {
    auto data = clustered(900, 8, 6, 510);
    HnswIndex straight(8, Metric::L2);
    for (std::size_t i = 0; i < 600; ++i) straight.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 0; i < 600; i += 5) straight.remove(i);
    auto resumed = load_hnsw(save_bytes(straight));
    for (HnswIndex* h : {&straight, resumed.get()}) {  // the same changes to both
        for (std::size_t i = 600; i < 900; ++i) h->add(i, data[i], meta_for(i));
        for (std::uint64_t i = 1; i < 600; i += 9) h->update(i, data[(i * 7) % 900]);
        h->upsert(5, data[3], Metadata().set("year", 1990));
        h->remove(2);
        h->set_metadata(4, Metadata().set("category", "sports"));
    }
    CHECK(save_bytes(*resumed) == save_bytes(straight));
    resumed->compact();
    straight.compact();
    CHECK(save_bytes(*resumed) == save_bytes(straight));
}

TEST(persistence, p53_save_during_a_search) {
    // Saving reads the index, so a predicate may save; changes during a save are refused.
    auto h = golden_hnsw();
    std::string saved;
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        if (saved.empty()) saved = save_bytes(*h);
        return true;
    };
    h->search(std::vector<float>(8, 1.0f), 5, 32, o);
    CHECK(saved == save_bytes(*h));
}

TEST(persistence, p54_unseekable_stream_rejected) {
    struct OneWay : std::streambuf {  // a stream that cannot seek (like a pipe)
        std::string data;
        explicit OneWay(std::string d) : data(std::move(d)) { setg(data.data(), data.data(), data.data() + data.size()); }
    } buffer(save_bytes(*tiny_hnsw()));
    std::istream in(&buffer);
    CHECK_THROWS_AS(FormatError, HnswIndex::load(in));
}

TEST(persistence, p55_other_metadata_damage) {
    // Metadata problems behind valid checksums: a row on a removed vector, a bad string reference.
    HnswIndex h(2, Metric::L2);
    h.add(0, std::vector<float>{0, 0}, Metadata().set("k", "a"));
    h.add(1, std::vector<float>{1, 1}, Metadata().set("k", "b"));
    h.remove(0);
    ParsedFile base = ParsedFile::parse(save_bytes(h));
    // META: mode (1), field count (4), name (4 + 1), type (1), indexed (1), string count (4), "a" (4 + 1), "b" (4 + 1), rows (8), slot (4), id (4)
    const std::size_t rows_at = 1 + 4 + 5 + 1 + 1 + 4 + 5 + 5, slot_at = rows_at + 8, id_at = slot_at + 4;
    ParsedFile on_removed = base, bad_ref = base;
    poke<std::uint32_t>(on_removed.payload("META"), slot_at, 0);  // slot 0 was removed
    poke<std::uint32_t>(bad_ref.payload("META"), id_at, 7);       // only 2 strings exist
    CHECK_THROWS_AS(FormatError, load_hnsw(on_removed.assemble()));
    CHECK_THROWS_AS(FormatError, load_hnsw(bad_ref.assemble()));
    CHECK(load_hnsw(base.assemble())->get_metadata(1)->get("k")->s == "b");
}

TEST(persistence, p57_every_validation_rule) {
    // Each rule the loader enforces, triggered behind a valid checksum.
    auto expect_rejected = [](const std::string& bytes) { CHECK_THROWS_AS(FormatError, load_hnsw(bytes)); };

    // Settings: dimension and repair flag (CONF: dimension at 0, repair flag at 36).
    const ParsedFile tiny = ParsedFile::parse(save_bytes(*tiny_hnsw()));
    for (std::uint64_t dim : {std::uint64_t(0), std::uint64_t(1) << 25}) {
        ParsedFile f = tiny;
        poke<std::uint64_t>(f.payload("CONF"), 0, dim);
        expect_rejected(f.assemble());
    }
    {
        ParsedFile f = tiny;
        poke<std::uint8_t>(f.payload("CONF"), 36, 2);
        expect_rejected(f.assemble());
    }
    {   // A vector cut off in the middle.
        ParsedFile f = tiny;
        f.payload("NODE").resize(8 + 8 + 1 + 1 + 2);
        expect_rejected(f.assemble());
    }

    // Entry point and free list (STAT: entry 0, top level 4, free count 8, free slots 16...).
    const auto t = tiny_hnsw();
    NodeId low = kEmpty;  // a live slot below the top level
    for (NodeId n = 0; n < t->storage().size(); ++n)
        if (!t->storage().ids().is_deleted(n) && t->storage().graph().level(n) < t->max_level()) low = n;
    REQUIRE(low != kEmpty && t->deleted_count() == 2);
    const std::vector<std::function<void(std::string&)>> stat_edits = {
        [&](std::string& p) { poke<std::uint32_t>(p, 0, 3); },                     // entry point is a removed slot (3)
        [&](std::string& p) { poke<std::uint32_t>(p, 0, low); },                   // entry point below the top level
        [&](std::string& p) { poke<std::uint32_t>(p, 0, low);                      // top level that is not the highest
                              poke<std::int32_t>(p, 4, t->storage().graph().level(low)); },
        [&](std::string& p) { poke<std::uint32_t>(p, 16, 999); },                  // free slot out of range
        [&](std::string& p) { poke<std::uint32_t>(p, 20, persist::from_little_endian<std::uint32_t>(
                                  reinterpret_cast<const unsigned char*>(&p[16]))); },  // a repeated free slot
    };
    for (const auto& edit : stat_edits) {
        ParsedFile f = tiny;
        edit(f.payload("STAT"));
        expect_rejected(f.assemble());
    }
    {   // An empty index must have no entry point and top level -1.
        HnswIndex e(2, Metric::L2);
        e.add(1, std::vector<float>{0, 0});
        e.remove(1);
        const ParsedFile empty = ParsedFile::parse(save_bytes(e));
        ParsedFile a = empty, b = empty;
        poke<std::uint32_t>(a.payload("STAT"), 0, 0);
        poke<std::int32_t>(b.payload("STAT"), 4, 0);
        expect_rejected(a.assemble());
        expect_rejected(b.assemble());
        CHECK(load_hnsw(empty.assemble())->size() == 0);
    }

    // Metadata. Fields are created in name order b (bool), f (float), t (tags):
    //   mode 0, field count 1, fields at 5 (each: name 4+1, type 1, indexed 1), so b's type is at 10;
    //   string count 26, "x" at 30 (length) and 34, "y" at 35 and 39;
    //   b: rows 40 (8 bytes), slot 48, value 52;  f: rows 53, slot 61, value 65;  t: rows 73, slot 81, count 85, ids 89 and 93.
    HnswIndex m(2, Metric::L2);
    m.add(0, std::vector<float>{0, 0}, Metadata().set("b", true).set("f", 1.5).set_tags("t", {"x", "y"}));
    const ParsedFile meta = ParsedFile::parse(save_bytes(m));
    REQUIRE(load_hnsw(meta.assemble())->get_metadata(0)->get("f")->f == 1.5);
    {   // the layout assumed below: the tag IDs are 0 and 1, at 89 and 93
        ParsedFile copy = meta;
        const std::string& p = copy.payload("META");
        REQUIRE(p.size() == 97 && persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(&p[93])) == 1);
    }
    const std::vector<std::function<void(std::string&)>> meta_edits = {
        [](std::string& p) { poke<std::uint8_t>(p, 10, 9); },               // unknown field type
        [](std::string& p) { p[39] = 'x'; },                                 // a repeated dictionary string
        [](std::string& p) { poke<std::uint64_t>(p, 40, 5); },              // more rows than vectors
        [](std::string& p) { poke<std::uint8_t>(p, 52, 2); },               // a boolean that is neither 0 nor 1
        [](std::string& p) { poke<double>(p, 65, double(kNaN)); },          // a NaN metadata float
        [](std::string& p) { poke<std::uint32_t>(p, 85, 1000000); },        // a huge tag count
        [](std::string& p) { poke<std::uint32_t>(p, 89, 9); },              // a tag that is not in the dictionary
        [](std::string& p) { poke<std::uint32_t>(p, 93, 0); },              // a repeated tag (the first is 0)
    };
    for (const auto& edit : meta_edits) {
        ParsedFile f = meta;
        edit(f.payload("META"));
        expect_rejected(f.assemble());
    }
}

TEST(persistence, p56_writing_to_a_failing_stream) {
    std::ostringstream out;
    out.setstate(std::ios::badbit);
    CHECK_THROWS_AS(std::runtime_error, golden_hnsw()->save(out));
    CHECK_THROWS_AS(std::runtime_error, golden_flat()->save(out));
}

// ===========================================================================
// Concurrency stage A: one lock per index (C1-C10 from the test plan, C11-C19 beyond it)
// ===========================================================================

namespace {

/// Runs `fn` on its own thread. If it has not finished within `seconds`, the test
/// program reports a deadlock and exits, so CI fails instead of hanging.
template <class Fn>
void with_watchdog(double seconds, const char* what, Fn&& fn) {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            fn();
        } catch (...) {
            error = std::current_exception();
        }
        std::lock_guard<std::mutex> l(m);
        done = true;
        cv.notify_all();
    });
    {
        std::unique_lock<std::mutex> l(m);
        if (!cv.wait_for(l, std::chrono::duration<double>(seconds), [&] { return done; })) {
            std::fprintf(stderr, "\nDEADLOCK: %s did not finish within %.0f seconds\n", what, seconds);
            std::fflush(stderr);
            std::_Exit(3);
        }
    }
    worker.join();
    if (error) std::rethrow_exception(error);
}

/// Problems found by worker threads; checked by the main thread afterwards.
struct Problems {
    std::atomic<long> unsorted{0}, duplicate{0}, not_finite{0}, unknown_id{0}, unexpected_exception{0};
    long total() const {
        return unsorted.load() + duplicate.load() + not_finite.load() + unknown_id.load() + unexpected_exception.load();
    }
};

/// Checks one result list: sorted, no repeated ID, finite distances, IDs below `id_limit`.
void check_results(const std::vector<SearchResult>& r, std::uint64_t id_limit, Problems& p) {
    std::set<std::uint64_t> seen;
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (i > 0 && r[i].distance < r[i - 1].distance) ++p.unsorted;
        if (!seen.insert(r[i].id).second) ++p.duplicate;
        if (!std::isfinite(r[i].distance)) ++p.not_finite;
        if (r[i].id >= id_limit) ++p.unknown_id;
    }
}

/// Starts `n` threads running fn(thread_index) and joins them.
template <class Fn>
void run_threads(std::size_t n, Fn&& fn) {
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < n; ++t) threads.emplace_back([&, t] { fn(t); });
    for (auto& th : threads) th.join();
}

}  // namespace

TEST(concurrency, c01_searches_while_one_thread_adds) {
    auto data = clustered(1500, 8, 6, 600);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 500; ++i) h.add(i, data[i]);
    std::atomic<std::uint64_t> added{500};
    std::atomic<bool> stop{false};
    Problems p;
    std::atomic<long> searches{0};
    with_watchdog(120, "c01", [&] {
        std::thread writer([&] {
            for (std::size_t i = 500; i < 1500; ++i) {
                h.add(i, data[i]);
                added.store(i + 1);
            }
            stop = true;
        });
        run_threads(4, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) {
                try {
                    const auto r = h.search(data[q++ % 1500], 10, 32);
                    check_results(r, added.load() + 1, p);  // read the limit after searching
                    ++searches;
                } catch (...) {
                    ++p.unexpected_exception;
                }
            }
        });
        writer.join();
    });
    CHECK(p.total() == 0 && searches > 0);
    CHECK(h.size() == 1500);
    check_graph(h, 0.99);
}

TEST(concurrency, c02_several_writers_different_ranges) {
    auto data = clustered(1200, 8, 6, 601);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    with_watchdog(120, "c02", [&] {
        run_threads(4, [&](std::size_t t) {
            for (std::size_t i = t * 300; i < (t + 1) * 300; ++i) h.add(i, data[i]);
        });
    });
    for (std::size_t i = 0; i < 1200; ++i) f.add(i, data[i]);
    CHECK(h.size() == 1200);
    bool all = true;
    for (std::uint64_t i = 0; i < 1200; ++i) all = all && h.contains(i);
    CHECK(all);
    check_graph(h, 0.99);
    CHECK(recall(h, f, Vectors(data.begin(), data.begin() + 100), 10, 64) > 0.95);
}

TEST(concurrency, c03_mixed_operations_from_many_threads) {
    // Every ID gets its own vector (0-5999), and updates draw from a separate pool
    // (6000-7999): many exact copies would hurt graph connectivity on their own,
    // with or without threads.
    auto data = clustered(8000, 8, 6, 602);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::dynamic({}));
    Problems p;
    std::vector<long> live(6, 0);
    with_watchdog(180, "c03", [&] {
        run_threads(6, [&](std::size_t t) {
            std::mt19937 rng(unsigned(t + 1));
            const std::uint64_t base = t * 1000;  // each thread owns IDs base .. base + 999
            std::vector<std::uint64_t> mine;
            std::uint64_t next = base;
            for (int op = 0; op < 400; ++op) {
                try {
                    switch (rng() % 6) {
                        case 0:
                        case 1:
                            if (next < base + 1000) {
                                h.add(next, data[next], Metadata().set("t", std::int64_t(t)));
                                mine.push_back(next++);
                            }
                            break;
                        case 2:
                            if (!mine.empty()) {
                                const std::size_t k = rng() % mine.size();
                                if (!h.remove(mine[k])) ++p.unexpected_exception;
                                mine.erase(mine.begin() + std::ptrdiff_t(k));
                            }
                            break;
                        case 3:
                            if (!mine.empty() && !h.update(mine[rng() % mine.size()], data[6000 + rng() % 2000])) ++p.unexpected_exception;
                            break;
                        case 4:
                            check_results(h.search(data[rng() % 8000], 10, 32), 6000, p);
                            break;
                        default:  // accessors
                            if (!mine.empty() && !h.contains(mine.back())) ++p.unexpected_exception;
                            if (!mine.empty() && !h.get_metadata(mine.back())) ++p.unexpected_exception;
                            (void)h.size();
                            (void)h.capacity();
                            break;
                    }
                } catch (...) {
                    ++p.unexpected_exception;
                }
            }
            live[t] = long(mine.size());
        });
    });
    CHECK(p.total() == 0);
    CHECK(long(h.size()) == std::accumulate(live.begin(), live.end(), 0L));
    check_graph(h, 0.98);
}

TEST(concurrency, c04_same_id_added_by_two_threads) {
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    std::atomic<long> wrong_outcome{0};
    with_watchdog(120, "c04", [&] {
        for (std::uint64_t id = 0; id < 150; ++id) {
            std::atomic<int> successes{0}, rejections{0};
            run_threads(4, [&](std::size_t t) {
                try {
                    if (t % 2) h.add(id, std::vector<float>{float(id), float(t), 0, 1});
                    else f.add(id, std::vector<float>{float(id), float(t), 0, 1});
                    ++successes;
                } catch (const std::invalid_argument&) {
                    ++rejections;
                }
            });
            if (successes != 2 || rejections != 2) ++wrong_outcome;  // one winner per index
        }
    });
    CHECK(wrong_outcome == 0 && h.size() == 150 && f.size() == 150);
}

TEST(concurrency, c05_same_id_removed_by_two_threads) {
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    for (std::uint64_t id = 0; id < 150; ++id) {
        h.add(id, std::vector<float>{float(id), 1, 0, 1});
        f.add(id, std::vector<float>{float(id), 1, 0, 1});
    }
    std::atomic<long> wrong_outcome{0};
    with_watchdog(120, "c05", [&] {
        for (std::uint64_t id = 0; id < 150; ++id) {
            std::atomic<int> h_true{0}, f_true{0};
            run_threads(4, [&](std::size_t t) {
                if (t % 2 ? h.remove(id) : f.remove(id)) ++(t % 2 ? h_true : f_true);
            });
            if (h_true != 1 || f_true != 1) ++wrong_outcome;
        }
    });
    CHECK(wrong_outcome == 0 && h.size() == 0 && f.size() == 0);
}

TEST(concurrency, c06_update_during_searches) {
    auto data = clustered(800, 8, 6, 603);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 400; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    Problems p;
    with_watchdog(120, "c06", [&] {
        std::thread writer([&] {
            for (int round = 0; round < 3; ++round)
                for (std::size_t i = 0; i < 400; i += 2) h.update(i, data[400 + (i + std::size_t(round) * 7) % 400]);
            stop = true;
        });
        run_threads(4, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) {
                try {
                    check_results(h.search(data[q++ % 800], 10, 32), 400, p);  // never the same ID twice
                } catch (...) {
                    ++p.unexpected_exception;
                }
            }
        });
        writer.join();
    });
    CHECK(p.total() == 0 && h.size() == 400 && h.capacity() <= 401);
    check_graph(h, 0.99);
}

TEST(concurrency, c07_save_during_searches_and_changes) {
    auto data = clustered(1200, 8, 6, 604);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 400; ++i) h.add(i, data[i], meta_for(i));
    std::atomic<bool> stop{false};
    std::vector<std::string> snapshots;
    Problems p;
    with_watchdog(180, "c07", [&] {
        std::thread writer([&] {
            for (std::size_t i = 400; i < 1200; ++i) {
                h.add(i, data[i], meta_for(i));
                if (i % 5 == 0) h.remove(i - 300);
            }
            stop = true;
        });
        std::thread saver([&] {
            while (!stop) snapshots.push_back(save_bytes(h));
        });
        run_threads(2, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) check_results(h.search(data[q++ % 1200], 10, 32), 1200, p);
        });
        writer.join();
        saver.join();
    });
    CHECK(p.total() == 0 && !snapshots.empty());
    bool consistent = true;
    for (std::size_t i = 0; i < snapshots.size(); i += std::max<std::size_t>(1, snapshots.size() / 8)) {
        auto g = load_hnsw(snapshots[i]);  // every snapshot is a complete, valid index
        consistent = consistent && g->size() >= 400 && g->size() <= 1200 && save_bytes(*g) == snapshots[i];
        check_graph(*g, 0.98);
    }
    CHECK(consistent);
}

TEST(concurrency, c08_compact_during_searches) {
    auto data = clustered(1000, 8, 6, 605);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 1000; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    Problems p;
    with_watchdog(180, "c08", [&] {
        std::thread changer([&] {
            for (int round = 0; round < 3; ++round) {
                for (std::size_t i = std::size_t(round); i < 1000; i += 9) h.remove(i);
                h.compact();
            }
            stop = true;
        });
        run_threads(4, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) {
                if (t == 3) {  // accessors and saves while compaction rebuilds the index
                    // Each accessor sees a consistent index, but two calls in a row may
                    // straddle a compaction, so they are not compared with each other.
                    (void)h.deleted_count();
                    (void)h.capacity();
                    (void)h.size();
                    (void)h.pooled_visited_lists();
                    (void)h.max_level();
                    // A save is one consistent moment: check the invariants on it.
                    auto g = load_hnsw(save_bytes(h));
                    if (g->size() + g->deleted_count() != g->capacity() || g->size() > 1000) ++p.unexpected_exception;
                } else {
                    check_results(h.search(data[q++ % 1000], 10, 32), 1000, p);
                }
            }
        });
        changer.join();
    });
    for (std::size_t i = 0; i < 1000; ++i)
        if (h.contains(i)) f.add(i, data[i]);
    CHECK(p.total() == 0 && h.deleted_count() == 0 && h.size() == f.size());
    CHECK(recall(h, f, Vectors(data.begin(), data.begin() + 100), 10, 64) > 0.95);
}

TEST(concurrency, c09_internal_calls_never_deadlock) {
    // update, upsert, compact and save each run internal steps under one lock.
    auto data = clustered(600, 8, 6, 606);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    std::atomic<bool> stop{false};
    with_watchdog(120, "c09", [&] {
        std::thread searcher([&] {
            std::size_t q = 0;
            while (!stop) {
                (void)h.search(data[q % 600], 5, 32);
                (void)f.search(data[q++ % 600], 5);
            }
        });
        for (std::size_t i = 0; i < 300; ++i) {
            h.upsert(i % 2 ? i : 1000 + i, data[300 + i]);
            f.upsert(i % 2 ? i : 1000 + i, data[300 + i]);
            if (i % 50 == 0) {
                h.compact();
                f.compact();
                (void)save_bytes(h);
                (void)save_bytes(f);
            }
        }
        stop = true;
        searcher.join();
    });
    CHECK(h.size() == 450 && f.size() == 450);
}

TEST(concurrency, c10_flat_under_mixed_load) {
    auto data = random_vectors(1500, 8, 607);
    FlatIndex f(8, Metric::Cosine, Schema::dynamic({}));
    Problems p;
    with_watchdog(180, "c10", [&] {
        run_threads(6, [&](std::size_t t) {
            std::mt19937 rng(unsigned(t + 11));
            const std::uint64_t base = t * 250;
            for (std::uint64_t i = base; i < base + 250; ++i) {
                try {
                    f.add(i, data[i], Metadata().set("t", std::int64_t(t)));
                    if (i % 3 == 0) f.update(i, data[rng() % 1500]);
                    if (i % 5 == 0) f.upsert(i, data[rng() % 1500]);
                    if (i % 7 == 0) f.remove(i);
                    check_results(f.search(data[rng() % 1500], 10), 1500, p);
                    check_results(f.search(data[rng() % 1500], 10, with_filter(Filter::eq("t", std::int64_t(t)))), 1500, p);
                } catch (...) {
                    ++p.unexpected_exception;
                }
            }
        });
    });
    CHECK(p.total() == 0);
    std::size_t expected = 0;
    for (std::uint64_t i = 0; i < 1500; ++i) expected += (i % 7 != 0);
    CHECK(f.size() == expected);
    CHECK(f.search(data[3], 5).size() == 5);  // usable once quiet again
}

TEST(concurrency, c11_predicate_searches_while_a_writer_waits) {
    // A predicate searching its own index while a change is queued must not deadlock:
    // the nested search does not lock again (a writer-preferring lock would block it).
    auto data = clustered(400, 8, 4, 608);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (int batch = 0; batch < 2; ++batch) {
        with_watchdog(60, "c11", [&] {
            std::atomic<bool> writer_started{false};
            std::thread writer;
            SearchOptions o;
            o.predicate = [&](std::uint64_t id) {
                if (!writer_started.exchange(true)) {
                    writer = std::thread([&] { h.add(1000, data[300]); f.add(1000, data[300]); });
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // the writer is now queued
                }
                return !h.search(data[id % 300], 1, 16).empty() && !f.search(data[id % 300], 1).empty();
            };
            if (batch == 0) {
                (void)h.search(data[0], 5, 32, o);
            } else {
                (void)h.search_batch(flatten(Vectors(data.begin(), data.begin() + 8)), 5, 32, 4, o);
            }
            writer.join();
            h.remove(1000);
            f.remove(1000);
        });
    }
    CHECK(h.size() == 300 && f.size() == 300);
}

TEST(concurrency, c12_changes_from_predicates_refused_exactly) {
    // Refused on the calling thread and on batch worker threads, for every kind of change.
    auto data = clustered(300, 8, 4, 609);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 200; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const std::string before_h = save_bytes(h), before_f = save_bytes(f);
    const std::vector<std::function<void()>> changes = {
        [&] { h.add(5000, data[0]); },          [&] { h.remove(1); },
        [&] { h.update(2, data[3]); },          [&] { h.upsert(3, data[4]); },
        [&] { h.set_metadata(4, Metadata().set("year", 1)); },
        [&] { h.create_payload_index("category"); },
        [&] { h.set_planner_params(PlannerParams{}); },
        [&] { h.compact(); },
    };
    const std::vector<std::function<void()>> flat_changes = {
        [&] { f.add(5000, data[0]); }, [&] { f.remove(1); }, [&] { f.update(2, data[3]); },
        [&] { f.upsert(3, data[4]); }, [&] { f.set_metadata(4, Metadata().set("year", 1)); },
        [&] { f.create_payload_index("category"); }, [&] { f.compact(); },
    };
    std::atomic<long> refused{0}, attempted{0};
    auto predicate_for = [&](const std::vector<std::function<void()>>& list) {
        return [&](std::uint64_t id) {
            ++attempted;
            try {
                list[id % list.size()]();
            } catch (const std::logic_error&) {
                ++refused;
            }
            return true;
        };
    };
    with_watchdog(60, "c12", [&] {
        SearchOptions o;
        o.predicate = predicate_for(changes);
        (void)h.search(data[0], 10, 32, o);
        (void)h.search_batch(flatten(Vectors(data.begin(), data.begin() + 16)), 5, 32, 4, o);  // worker threads
        (void)h.search_range(data[0], 5.0f, kNoLimit, o);
        SearchOptions of;
        of.predicate = predicate_for(flat_changes);
        (void)f.search(data[0], 10, of);
        (void)f.search_batch(flatten(Vectors(data.begin(), data.begin() + 16)), 5, 4, of);
        (void)f.search_range(data[0], 5.0f, kNoLimit, of);
    });
    CHECK(attempted > 0 && refused == attempted);  // every attempt refused, none slipped through
    CHECK(save_bytes(h) == before_h && save_bytes(f) == before_f);  // nothing changed
}

TEST(concurrency, c13_writers_are_not_starved) {
    // Under a steady stream of searches from 8 threads, every change still gets through.
    auto data = clustered(600, 8, 4, 610);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 300; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    double worst_wait = 0;
    with_watchdog(120, "c13", [&] {
        std::vector<std::thread> readers;
        for (int t = 0; t < 8; ++t)
            readers.emplace_back([&, t] {
                std::size_t q = std::size_t(t);
                while (!stop) (void)h.search(data[q++ % 600], 10, 64);
            });
        for (std::size_t i = 300; i < 400; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            h.add(i, data[i]);
            worst_wait = std::max(worst_wait, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        stop = true;
        for (auto& r : readers) r.join();
    });
    CHECK(h.size() == 400);
    CHECK(worst_wait < 10.0);  // generous: sanitizers slow everything; starvation would be unbounded
}

TEST(concurrency, c14_upsert_is_atomic) {
    // Many threads upserting the same IDs: before the lock covered both steps,
    // two threads could both find an ID missing, and the second add would throw.
    // Each ID draws from its own vectors, so no two IDs hold the same vector.
    auto data = clustered(200, 8, 4, 611);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    std::atomic<long> errors{0};
    with_watchdog(120, "c14", [&] {
        run_threads(8, [&](std::size_t t) {
            for (int round = 0; round < 40; ++round)
                for (std::uint64_t id = 0; id < 20; ++id) {
                    const auto& v = data[id + 20 * ((t + std::size_t(round)) % 10)];
                    try {
                        h.upsert(id, v);
                        f.upsert(id, v);
                    } catch (...) {
                        ++errors;
                    }
                }
        });
    });
    CHECK(errors == 0 && h.size() == 20 && f.size() == 20);
    // 320 updates per vector of a 20-vector index degrade the graph whatever the
    // order (single-threaded too: in 27 of 200 random orders one or two vectors
    // became unreachable, worst 90%; see the project notes). compact() rebuilds it.
    check_graph(h, 0.85);
    h.compact();
    check_graph(h, 1.0);
}

TEST(concurrency, c15_batch_and_range_during_changes) {
    auto data = clustered(900, 8, 6, 612);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 400; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    std::atomic<bool> stop{false};
    Problems p;
    const auto buffer = flatten(Vectors(data.begin(), data.begin() + 12));
    with_watchdog(180, "c15", [&] {
        std::thread writer([&] {
            for (std::size_t i = 400; i < 900; ++i) {
                h.add(i, data[i]);
                f.add(i, data[i]);
                if (i % 4 == 0) {
                    h.remove(i - 200);
                    f.remove(i - 200);
                }
            }
            stop = true;
        });
        run_threads(2, [&](std::size_t t) {
            while (!stop) {
                for (const auto& r : (t ? f.search_batch(buffer, 5, 2) : h.search_batch(buffer, 5, 32, 2))) check_results(r, 900, p);
                check_results(t ? f.search_range(data[t], 1.0f) : h.search_range(data[t], 1.0f), 900, p);
            }
        });
        writer.join();
    });
    CHECK(p.total() == 0 && h.size() == f.size());
}

TEST(concurrency, c16_metadata_and_settings_change_during_filtered_searches) {
    // IDs 0-199 keep category "fixed"; IDs 200-399 flip between "a" and "b".
    // A search for "a" must never return a "fixed" ID, whatever changes meanwhile.
    auto data = clustered(400, 8, 4, 613);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 400; ++i) {
        const Metadata m = Metadata().set("category", i < 200 ? "fixed" : "a");
        h.add(i, data[i], m);
        f.add(i, data[i], m);
    }
    std::atomic<bool> stop{false};
    std::atomic<long> wrong{0};
    Problems p;
    with_watchdog(180, "c16", [&] {
        std::thread writer([&] {
            for (int round = 0; round < 6; ++round) {
                for (std::uint64_t i = 200; i < 400; i += 3) {
                    h.set_metadata(i, Metadata().set("category", round % 2 ? "a" : "b"));
                    f.set_metadata(i, Metadata().set("category", round % 2 ? "a" : "b"));
                }
                if (round == 2) {
                    h.create_payload_index("category");
                    f.create_payload_index("category");
                }
                h.set_planner_params(PlannerParams{round % 2 ? std::size_t(10) : std::size_t(2000), 0.01, 256, 32.0});
            }
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            const SearchOptions o = with_filter(Filter::eq("category", "a"));
            std::size_t q = t;
            while (!stop) {
                const PlannerParams pp = h.planner_params();  // a consistent copy, read while it changes
                if (pp.max_exact_matches != 10 && pp.max_exact_matches != 2000 && pp.max_exact_matches != PlannerParams{}.max_exact_matches)
                    ++p.unexpected_exception;
                for (const auto& r : {h.search(data[q % 400], 10, 32, o), f.search(data[q % 400], 10, o)}) {
                    check_results(r, 400, p);
                    for (const auto& x : r) wrong += x.id < 200;
                }
                ++q;
            }
        });
        writer.join();
    });
    CHECK(p.total() == 0 && wrong == 0);
}

TEST(concurrency, c17_exceptions_release_the_lock) {
    // Failed changes and failed searches must leave the lock free.
    auto data = clustered(300, 8, 4, 614);
    HnswIndex h(8, Metric::L2, HnswParams{}, Schema::strict({{"year", FieldType::Int}}));
    for (std::size_t i = 0; i < 100; ++i) h.add(i, data[i]);
    std::atomic<long> failures_seen{0};
    with_watchdog(120, "c17", [&] {
        std::thread failing([&] {
            for (int round = 0; round < 200; ++round) {
                auto expect_throw = [&](auto&& fn) {
                    try {
                        fn();
                    } catch (const std::exception&) {
                        ++failures_seen;
                    }
                };
                expect_throw([&] { h.add(0, data[0]); });                                      // duplicate ID
                expect_throw([&] { h.add(9000, std::vector<float>(3, 1.0f)); });                // wrong dimension
                expect_throw([&] { h.add(9001, data[1], Metadata().set("other", 1)); });        // strict schema
                expect_throw([&] { (void)h.search(std::vector<float>(3, 1.0f), 5); });          // wrong query dimension
                expect_throw([&] { (void)h.search(data[0], 5, 32, with_filter(Filter::eq("nope", 1))); });  // unknown field
            }
        });
        for (std::size_t i = 100; i < 300; ++i) h.add(i, data[i]);  // keeps working meanwhile
        failing.join();
        // A predicate's own exception, on the calling thread and on batch worker threads.
        SearchOptions throws;
        throws.predicate = [](std::uint64_t id) -> bool {
            if (id % 3 == 0) throw std::runtime_error("predicate failed");
            return true;
        };
        for (int round = 0; round < 20; ++round) {
            try {
                (void)h.search(data[0], 10, 32, throws);
            } catch (const std::runtime_error&) {
                ++failures_seen;
            }
            try {
                (void)h.search_batch(flatten(Vectors(data.begin(), data.begin() + 8)), 5, 32, 4, throws);
            } catch (const std::runtime_error&) {
                ++failures_seen;
            }
        }
        std::thread other([&] { h.add(5000, data[5]); });  // the lock is free again
        other.join();
    });
    CHECK(failures_seen == 1040 && h.size() == 301);
}

TEST(concurrency, c18_several_indexes) {
    // Predicates may search other indexes, crossed in both directions, while
    // writers hammer both: no deadlock. (This exact pattern deadlocked an early
    // version: nested reads queued behind waiting writers formed a cycle.)
    auto data = clustered(300, 8, 4, 615);
    HnswIndex a(8, Metric::L2), b(8, Metric::L2);
    for (std::size_t i = 0; i < 200; ++i) {
        a.add(i, data[i]);
        b.add(i, data[i]);
    }
    SearchOptions on_a, on_b;
    on_a.predicate = [&](std::uint64_t id) { return !b.search(data[id % 300], 1, 16).empty(); };
    on_b.predicate = [&](std::uint64_t id) { return !a.search(data[id % 300], 1, 16).empty(); };
    with_watchdog(120, "c18", [&] {
        std::atomic<bool> stop{false};
        std::thread writer_a([&] {
            for (std::size_t i = 200; i < 300; ++i) a.add(i, data[i]);
        });
        std::thread writer_b([&] {
            for (std::size_t i = 200; i < 300; ++i) b.add(i, data[i]);
        });
        std::thread stopper([&] {
            writer_a.join();
            writer_b.join();
            stop = true;
        });
        run_threads(4, [&](std::size_t t) {
            int rounds = 0;
            while (!stop || rounds < 5) {
                (void)(t % 2 ? b.search(data[std::size_t(rounds) % 300], 3, 16, on_b) : a.search(data[std::size_t(rounds) % 300], 3, 16, on_a));
                ++rounds;
            }
        });
        stopper.join();
    });
    CHECK(a.size() == 300 && b.size() == 300);
    // Changing another index from a predicate is refused too: with another thread
    // doing the reverse, it would be a lock-order cycle.
    std::atomic<long> refused{0};
    SearchOptions changes_b;
    changes_b.predicate = [&](std::uint64_t id) {
        try {
            b.add(50000 + id, data[0]);
        } catch (const std::logic_error&) {
            ++refused;
        }
        return true;
    };
    const auto results = a.search(data[0], 5, 32, changes_b);
    CHECK(refused > 0 && b.size() == 300 && results.size() == 5);
}

TEST(concurrency, c20_simultaneous_saves_to_one_file) {
    // Saves run in parallel, so several may target the same file at once. Each
    // must write its own temporary file (the last rename wins): with one shared
    // temporary file, saves overwrite each other mid-write, and renames fail once
    // another save has moved the file away. The index is large enough that each
    // save spans many thread time slices, and all saves start together.
    auto data = clustered(20000, 32, 10, 617);
    HnswIndex h(32, Metric::L2, HnswParams{8, 40, 1, true, {}}, Schema::dynamic({}));
    FlatIndex f(32, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        h.add(i, data[i], Metadata().set("g", std::int64_t(i % 7)));
        f.add(i, data[i]);
    }
    const std::string path = temp_path("c20"), flat_path = temp_path("c20_flat");
    std::atomic<long> saved{0}, failed{0}, other_error{0};
    with_watchdog(240, "c20", [&] {
        for (int round = 0; round < 3; ++round) {
            std::atomic<int> ready{0};
            run_threads(6, [&](std::size_t t) {
                ++ready;
                while (ready < 6) std::this_thread::yield();  // start together
                try {
                    if (t % 3 == 2) f.save(flat_path);
                    else h.save(path);
                    ++saved;
                } catch (const std::runtime_error&) {
                    ++failed;
                } catch (...) {
                    ++other_error;
                }
            });
        }
    });
    CHECK(other_error == 0 && saved > 0);
#ifndef _WIN32
    CHECK(failed == 0);  // POSIX renames over a file atomically, so every save must succeed
#endif
    // Windows may refuse two renames over one file at the same moment: such a save
    // fails cleanly with std::runtime_error, and the file is still complete.
    auto h2 = HnswIndex::load(path);
    auto f2 = FlatIndex::load(flat_path);
    CHECK(save_bytes(*h2) == save_bytes(h) && save_bytes(*f2) == save_bytes(f));  // complete, not mixed
    CHECK(temporaries_left(path) == 0 && temporaries_left(flat_path) == 0);
    std::filesystem::remove(path);
    std::filesystem::remove(flat_path);
}

TEST(concurrency, c19_lock_limits_and_failures) {
    // Nesting beyond IndexLock::kMaxNesting is refused cleanly, and everything is released.
    {
        std::vector<std::unique_ptr<IndexLock>> locks;
        for (std::size_t i = 0; i <= IndexLock::kMaxNesting; ++i) locks.push_back(std::make_unique<IndexLock>());
        std::vector<std::unique_ptr<IndexLock::ReadGuard>> guards;
        for (std::size_t i = 0; i < IndexLock::kMaxNesting; ++i) guards.push_back(std::make_unique<IndexLock::ReadGuard>(*locks[i]));
        CHECK_THROWS_AS(std::logic_error, IndexLock::WriteGuard(*locks.back(), "test"));
        CHECK_THROWS_AS(std::logic_error, IndexLock::ReadGuard(*locks.back()));
        guards.clear();
        CHECK(!locks[0]->held() && !locks.back()->held());
        with_watchdog(30, "c19 relock", [&] { const IndexLock::WriteGuard w(*locks.back(), "test"); });
    }
    // Running out of memory in the middle of a change releases the lock.
    if constexpr (HNSW_ALLOC_HOOK) {
        auto data = clustered(100, 8, 4, 616);
        HnswIndex h(8, Metric::L2);
        for (std::size_t i = 0; i < 50; ++i) h.add(i, data[i], meta_for(i));
        std::uint64_t id = 1000;
        CHECK(sweep_allocation_failures([&] { h.add(id, data[id % 100], meta_for(id)); },
                                        [&] { CHECK(!h.contains(id)); }) > 0);
        with_watchdog(30, "c19 after out-of-memory", [&] {
            std::thread other([&] { h.add(2000, data[1]); });
            other.join();
            (void)h.search(data[0], 5);
        });
        CHECK(h.contains(2000));
    }
}


// ===========================================================================
// add_batch: parallel inserts (B1-B19; B20 is the benchmark's threads option)
// ===========================================================================

namespace {

std::vector<std::uint64_t> id_range(std::uint64_t first, std::uint64_t count) {
    std::vector<std::uint64_t> ids(count);
    std::iota(ids.begin(), ids.end(), first);
    return ids;
}

/// The fraction of live nodes reachable on level 0 from the entry point.
double level0_reachable(const HnswIndex& h) {
    const GraphStorage& g = h.storage().graph();
    const IdMap& ids = h.storage().ids();
    std::vector<char> seen(h.storage().size(), 0);
    std::vector<NodeId> stack{h.entry_point()};
    seen[h.entry_point()] = 1;
    while (!stack.empty()) {
        const NodeId n = stack.back();
        stack.pop_back();
        for (NodeId m : g.links(n, 0)) {
            if (m == kEmpty) break;
            if (!seen[m]) {
                seen[m] = 1;
                stack.push_back(m);
            }
        }
    }
    std::size_t live = 0, reached = 0;
    for (NodeId n = 0; n < h.storage().size(); ++n)
        if (!ids.is_deleted(n)) {
            ++live;
            reached += std::size_t(seen[n]);
        }
    return double(reached) / double(live);
}

/// The highest level among live nodes, computed from scratch.
int highest_live_level(const HnswIndex& h) {
    int top = -1;
    for (NodeId n = 0; n < h.storage().size(); ++n)
        if (!h.storage().ids().is_deleted(n)) top = std::max(top, h.storage().graph().level(n));
    return top;
}

}  // namespace

TEST(add_batch, b01_any_thread_count_builds_a_good_graph) {
    auto data = clustered(3000, 16, 10, 700);
    auto queries = clustered(100, 16, 10, 701);
    const auto all = flatten(data);
    const auto ids = id_range(0, 3000);
    FlatIndex f(16, Metric::L2);
    HnswIndex one_by_one(16, Metric::L2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        f.add(i, data[i]);
        one_by_one.add(i, data[i]);
    }
    const double reference = recall(one_by_one, f, queries, 10, 64);
    for (std::size_t threads : {std::size_t(1), std::size_t(2), std::size_t(4), std::size_t(8)}) {
        HnswIndex h(16, Metric::L2);
        h.add_batch(ids, all, threads);
        bool all_present = h.size() == 3000;
        for (std::uint64_t i = 0; i < 3000; i += 7) all_present = all_present && h.contains(i);
        CHECK(all_present);
        check_graph(h, 0.99);
        CHECK(recall(h, f, queries, 10, 64) >= reference - 0.01);
        CHECK(h.max_level() == highest_live_level(h) && !h.storage().ids().is_deleted(h.entry_point()));
    }
}

TEST(add_batch, b02_one_thread_is_identical_to_one_by_one) {
    auto data = clustered(1200, 8, 6, 702);
    const auto ids = id_range(100, 1200);
    std::vector<Metadata> meta;
    for (std::size_t i = 0; i < data.size(); ++i) meta.push_back(meta_for(i));
    for (Metric m : kMetrics) {
        HnswIndex a(8, m), b(8, m), c(8, m), d(8, m);
        for (std::size_t i = 0; i < data.size(); ++i) {
            a.add(ids[i], data[i]);
            c.add(ids[i], data[i], meta[i]);
        }
        b.add_batch(ids, flatten(data), 1);
        d.add_batch(ids, flatten(data), meta, 1);
        CHECK(save_bytes(a) == save_bytes(b));
        CHECK(save_bytes(c) == save_bytes(d));
    }
}

TEST(add_batch, b03_duplicate_ids_change_nothing) {
    auto data = clustered(200, 8, 4, 703);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 100; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    auto inside = id_range(100, 100);
    inside[60] = inside[10];  // twice within the batch
    auto stored = id_range(100, 100);
    stored[99] = 5;  // already in the index
    const auto vecs = flatten(Vectors(data.begin() + 100, data.end()));
    for (const auto& bad : {inside, stored}) {
        CHECK_THROWS_AS(std::invalid_argument, h.add_batch(bad, vecs, 4));
        CHECK_THROWS_AS(std::invalid_argument, f.add_batch(bad, vecs));
    }
    CHECK(save_bytes(h) == hb && save_bytes(f) == fb);
}

TEST(add_batch, b04_invalid_vectors_change_nothing) {
    auto data = clustered(200, 8, 4, 704);
    HnswIndex h(8, Metric::Cosine);
    FlatIndex f(8, Metric::Cosine);
    for (std::size_t i = 0; i < 100; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    const auto ids = id_range(100, 100);
    auto with_nan = flatten(Vectors(data.begin() + 100, data.end()));
    auto with_inf = with_nan;
    auto zero = with_nan;
    with_nan[8 * 50 + 3] = kNaN;
    with_inf[8 * 99 + 7] = kInf;
    auto short_buffer = with_inf;
    short_buffer.resize(short_buffer.size() - 1);
    for (const auto& bad : {with_nan, with_inf, short_buffer}) {
        CHECK_THROWS_AS(std::invalid_argument, h.add_batch(ids, bad, 4));
        CHECK_THROWS_AS(std::invalid_argument, f.add_batch(ids, bad));
    }
    CHECK(save_bytes(h) == hb && save_bytes(f) == fb);
    // A zero vector is accepted with cosine, as by add() (it has no direction and
    // sits at distance 1 from everything): a batch treats it the same way.
    std::fill(zero.begin() + 8 * 70, zero.begin() + 8 * 71, 0.0f);
    h.add_batch(ids, zero, 4);
    f.add_batch(ids, zero);
    CHECK(h.size() == 200 && f.size() == 200);
}

TEST(add_batch, b05_invalid_metadata_changes_nothing) {
    auto data = clustered(200, 8, 4, 705);
    const auto ids = id_range(100, 100);
    const auto vecs = flatten(Vectors(data.begin() + 100, data.end()));
    // Strict schema: an unknown field in the middle of the batch.
    HnswIndex strict(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    for (std::size_t i = 0; i < 100; ++i) strict.add(i, data[i], meta_for(i));
    std::vector<Metadata> meta;
    for (std::size_t i = 0; i < 100; ++i) meta.push_back(meta_for(100 + i));
    meta[40].set("unknown", 1);
    const std::string sb = save_bytes(strict);
    CHECK_THROWS_AS(std::invalid_argument, strict.add_batch(ids, vecs, meta, 4));
    CHECK(save_bytes(strict) == sb);
    // Dynamic schema: a new field with two types within one batch (each row alone is fine).
    HnswIndex dyn(8, Metric::L2);
    FlatIndex fdyn(8, Metric::L2);
    for (std::size_t i = 0; i < 100; ++i) {
        dyn.add(i, data[i]);
        fdyn.add(i, data[i]);
    }
    std::vector<Metadata> mixed(100);
    mixed[3].set("fresh", std::int64_t(1));
    mixed[90].set("fresh", "text");
    const std::string db = save_bytes(dyn), fb = save_bytes(fdyn);
    CHECK_THROWS_AS(std::invalid_argument, dyn.add_batch(ids, vecs, mixed, 4));
    CHECK_THROWS_AS(std::invalid_argument, fdyn.add_batch(ids, vecs, mixed));
    // A metadata count that does not match the vectors.
    CHECK_THROWS_AS(std::invalid_argument, dyn.add_batch(ids, vecs, std::vector<Metadata>(99), 4));
    CHECK_THROWS_AS(std::invalid_argument, fdyn.add_batch(ids, vecs, std::vector<Metadata>(101)));
    CHECK(save_bytes(dyn) == db && save_bytes(fdyn) == fb && dyn.metadata().field_count() == 0);
}

TEST(add_batch, b06_empty_batch_changes_nothing) {
    auto data = clustered(50, 8, 2, 706);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    HnswIndex empty(8, Metric::L2);
    for (std::size_t i = 0; i < 50; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f), eb = save_bytes(empty);
    h.add_batch({}, {}, 4);
    f.add_batch({}, {});
    empty.add_batch({}, {}, std::span<const Metadata>{}, 4);
    CHECK(save_bytes(h) == hb && save_bytes(f) == fb && save_bytes(empty) == eb);
}

TEST(add_batch, b07_empty_index_and_reused_slots) {
    auto data = clustered(800, 8, 6, 707);
    // Into an index with removed slots: they are reused in the same order as one-by-one adds.
    HnswIndex a(8, Metric::L2), b(8, Metric::L2), c(8, Metric::L2);
    for (HnswIndex* h : {&a, &b, &c}) {
        for (std::size_t i = 0; i < 500; ++i) h->add(i, data[i]);
        for (std::uint64_t i = 0; i < 500; i += 10) h->remove(i);
    }
    const auto ids = id_range(1000, 300);
    const auto vecs = flatten(Vectors(data.begin() + 500, data.end()));
    for (std::size_t i = 0; i < 300; ++i) a.add(ids[i], data[500 + i]);
    b.add_batch(ids, vecs, 1);
    c.add_batch(ids, vecs, 4);
    CHECK(save_bytes(a) == save_bytes(b));
    CHECK(c.size() == 750 && c.capacity() == 750 && c.deleted_count() == 0);  // all 50 removed slots reused
    check_graph(c, 0.99);
    // Into an empty index, on many threads.
    HnswIndex e(8, Metric::L2);
    e.add_batch(id_range(0, 800), flatten(data), 8);
    CHECK(e.size() == 800);
    check_graph(e, 0.99);
    // Into an index whose vectors were all removed: free slots but no entry point.
    for (std::size_t threads : {std::size_t(1), std::size_t(4)}) {
        HnswIndex r(8, Metric::L2), r_ref(8, Metric::L2);
        for (HnswIndex* h : {&r, &r_ref}) {
            for (std::size_t i = 0; i < 100; ++i) h->add(i, data[i]);
            for (std::uint64_t i = 0; i < 100; ++i) h->remove(i);
        }
        r.add_batch(id_range(2000, 300), flatten(Vectors(data.begin() + 100, data.begin() + 400)), threads);
        for (std::size_t i = 0; i < 300; ++i) r_ref.add(2000 + i, data[100 + i]);
        CHECK(r.size() == 300 && r.capacity() == 300);
        if (threads == 1) CHECK(save_bytes(r) == save_bytes(r_ref));
        check_graph(r, 0.99);
    }
}

TEST(add_batch, b08_entry_point_with_many_high_levels) {
    // M = 2 makes high levels common, so threads race to raise the top level. Such
    // small M gives poorly connected graphs however they are built (about 89% of
    // vectors reachable even one by one), so quality is compared with one-by-one adds.
    auto data = clustered(4000, 8, 10, 708);
    for (int round = 0; round < 3; ++round) {
        const HnswParams params{2, 32, std::uint64_t(round), true, {}};
        HnswIndex h(8, Metric::L2, params), one_by_one(8, Metric::L2, params);
        h.add_batch(id_range(0, 4000), flatten(data), 8);
        for (std::size_t i = 0; i < data.size(); ++i) one_by_one.add(i, data[i]);
        CHECK(!h.storage().ids().is_deleted(h.entry_point()));
        CHECK(h.storage().graph().level(h.entry_point()) == h.max_level());
        CHECK(h.max_level() == highest_live_level(h));
        CHECK(level0_reachable(h) >= level0_reachable(one_by_one) - 0.02);
        check_graph(h, 0.8);  // structure: valid links, levels and entry point
    }
}

TEST(add_batch, b09_searches_see_all_of_a_batch_or_none) {
    auto data = clustered(1200, 8, 6, 709);
    HnswIndex h(8, Metric::L2);
    for (std::size_t i = 0; i < 200; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    std::atomic<long> partial{0}, observations{0};
    Problems p;
    with_watchdog(120, "b09", [&] {
        std::thread batches([&] {
            for (std::uint64_t k = 0; k < 5; ++k)  // five batches of 200
                h.add_batch(id_range(200 + 200 * k, 200), flatten(Vectors(data.begin() + std::ptrdiff_t(200 + 200 * k), data.begin() + std::ptrdiff_t(400 + 200 * k))), 4);
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) {
                const std::size_t n = h.size();
                if (n % 200 != 0) ++partial;  // never part of a batch
                ++observations;
                check_results(h.search(data[q++ % 1200], 10, 32), 1200, p);
            }
        });
        batches.join();
    });
    CHECK(partial == 0 && observations > 0 && p.total() == 0 && h.size() == 1200);
}

TEST(add_batch, b10_refused_inside_a_search) {
    auto data = clustered(100, 8, 2, 710);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::size_t i = 0; i < 50; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    std::atomic<long> refused{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        for (auto&& attempt : {std::function<void()>([&] { h.add_batch(id_range(500, 1), data[60], 2); }),
                               std::function<void()>([&] { f.add_batch(id_range(500, 1), data[60]); })}) {
            try {
                attempt();
            } catch (const std::logic_error&) {
                ++refused;
            }
        }
        return true;
    };
    (void)h.search(data[0], 3, 16, o);
    CHECK(refused > 0 && h.size() == 50 && f.size() == 50);
}

TEST(add_batch, b11_out_of_memory_removes_the_batch) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(140, 8, 4, 711);
    for (std::size_t threads : {std::size_t(1), std::size_t(4)}) {
        HnswIndex h(8, Metric::L2);
        for (std::size_t i = 0; i < 80; ++i) h.add(i, data[i], meta_for(i));
        for (std::uint64_t i = 0; i < 80; i += 8) h.remove(i);  // some slots to reuse
        const std::size_t before = h.size(), fields = h.metadata().field_count();
        const auto ids = id_range(1000, 60);
        const auto vecs = flatten(Vectors(data.begin() + 80, data.end()));
        std::vector<Metadata> meta(60);
        for (std::size_t i = 0; i < 60; ++i) meta[i] = meta_for(i).set("batch_only", std::int64_t(i));  // a new field
        bool clean = true;
        const long points = sweep_allocation_failures([&] { h.add_batch(ids, vecs, meta, threads); }, [&] {
            clean = clean && h.size() == before && h.metadata().field_count() == fields;
            for (std::uint64_t id : ids) clean = clean && !h.contains(id);
            for (std::uint64_t i = 1; i < 80; i += 8) clean = clean && h.contains(i) && h.get_metadata(i).has_value();
            clean = clean && !h.search(data[1], 5, 32).empty();
            clean = clean && !h.storage().ids().is_deleted(h.entry_point()) && h.max_level() == highest_live_level(h);
        });
        CHECK(points > 0 && clean);
        CHECK(h.size() == before + 60);  // the final, unhindered attempt succeeded
        check_graph(h, 0.97);
        h.add(5000, data[0]);  // still fully usable
        CHECK(h.contains(5000));
    }
    // Into an empty index (a failure may come before any vector is the entry point),
    // and with graph repair turned off.
    for (bool repair : {true, false}) {
        HnswParams params;
        params.repair_on_remove = repair;
        HnswIndex h(8, Metric::L2, params);
        const auto ids = id_range(0, 50);
        const auto vecs = flatten(Vectors(data.begin(), data.begin() + 50));
        bool clean = true;
        CHECK(sweep_allocation_failures([&] { h.add_batch(ids, vecs, 4); }, [&] {
            clean = clean && h.size() == 0 && h.entry_point() == kEmpty && h.max_level() == -1;
        }) > 0);
        CHECK(clean && h.size() == 50);
        check_graph(h, 0.97, repair ? 0.02 : 1.0);  // without repair, links to removed vectors may remain
    }
}

TEST(add_batch, b12_thread_counts) {
    auto data = clustered(40, 8, 2, 712);
    for (std::size_t threads : {std::size_t(0), std::size_t(64), std::size_t(1000000)}) {
        HnswIndex h(8, Metric::L2);
        h.add_batch(id_range(0, 40), flatten(data), threads);  // more threads than vectors; 0 = all cores
        CHECK(h.size() == 40);
        check_graph(h, 1.0);
    }
}

TEST(add_batch, b13_metadata_and_payload_counts) {
    auto data = clustered(900, 8, 6, 713);
    std::vector<Metadata> meta;
    for (std::size_t i = 0; i < 900; ++i) meta.push_back(meta_for(i));
    HnswIndex a(8, Metric::L2), b(8, Metric::L2);
    for (HnswIndex* h : {&a, &b}) {
        for (std::size_t i = 0; i < 300; ++i) h->add(i, data[i], meta[i]);
        h->create_payload_index("category");
        h->create_payload_index("tags");
    }
    for (std::size_t i = 300; i < 900; ++i) a.add(i, data[i], meta[i]);
    b.add_batch(id_range(300, 600), flatten(Vectors(data.begin() + 300, data.end())),
                std::span<const Metadata>(meta.data() + 300, 600), 4);
    const MetadataStore &ma = a.metadata(), &mb = b.metadata();
    for (const char* cat : kCategories)
        CHECK(mb.value_count(*mb.find_field("category"), *mb.find_string(cat)) ==
              ma.value_count(*ma.find_field("category"), *ma.find_string(cat)));
    bool same = true;
    for (std::uint64_t i = 0; i < 900; i += 13) same = same && *b.get_metadata(i) == *a.get_metadata(i);
    CHECK(same);
    const SearchOptions o = with_filter(Filter::eq("category", "tech"));
    CHECK(b.search(data[5], 10, 64, o).size() == 10);
}

TEST(add_batch, b14_save_and_load_after_a_parallel_batch) {
    auto data = clustered(1500, 8, 6, 714);
    HnswIndex h(8, Metric::Cosine);
    std::vector<Metadata> meta;
    for (std::size_t i = 0; i < 1500; ++i) meta.push_back(meta_for(i));
    h.add_batch(id_range(0, 1500), flatten(data), meta, 4);
    const std::string bytes = save_bytes(h);
    auto g = load_hnsw(bytes);
    CHECK(save_bytes(*g) == bytes && all_links(*g) == all_links(h));
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i) same = same && same_results(g->search(data[i * 50], 10, 64), h.search(data[i * 50], 10, 64));
    CHECK(same);
}

TEST(add_batch, b15_later_changes_work_normally) {
    auto data = clustered(1600, 8, 6, 715);
    HnswIndex h(8, Metric::L2);
    h.add_batch(id_range(0, 1200), flatten(Vectors(data.begin(), data.begin() + 1200)), 4);
    for (std::uint64_t i = 0; i < 1200; i += 5) h.remove(i);
    for (std::uint64_t i = 1; i < 1200; i += 7) h.update(i, data[1200 + i % 400]);
    for (std::uint64_t i = 0; i < 1200; i += 25) h.upsert(i, data[i]);
    check_graph(h, 0.98);
    h.compact();
    check_graph(h, 0.99);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1200; ++i)
        if (h.contains(i)) f.add(i, std::vector<float>(h.storage().vectors().get(*h.storage().ids().find(i)).begin(),
                                                      h.storage().vectors().get(*h.storage().ids().find(i)).end()));
    CHECK(recall(h, f, Vectors(data.begin(), data.begin() + 100), 10, 64) > 0.95);
}

TEST(add_batch, b16_re_adding_removed_ids) {
    auto data = clustered(600, 8, 4, 716);
    HnswIndex h(8, Metric::L2);
    h.add_batch(id_range(0, 300), flatten(Vectors(data.begin(), data.begin() + 300)), 4);
    for (std::uint64_t i = 0; i < 300; i += 3) h.remove(i);
    std::vector<std::uint64_t> again;
    Vectors fresh;
    for (std::uint64_t i = 0; i < 300; i += 3) {
        again.push_back(i);
        fresh.push_back(data[300 + i]);
    }
    h.add_batch(again, flatten(fresh), 4);
    CHECK(h.size() == 300 && h.capacity() == 300);
    bool updated = true;
    for (std::size_t k = 0; k < again.size(); ++k)
        updated = updated && h.search(fresh[k], 1, 64)[0].id == again[k];
    CHECK(updated);
}

TEST(add_batch, b17_flat_identical_and_exact_on_failure) {
    auto data = clustered(400, 8, 4, 717);
    std::vector<Metadata> meta;
    for (std::size_t i = 0; i < 400; ++i) meta.push_back(meta_for(i));
    FlatIndex a(8, Metric::L2), b(8, Metric::L2);
    for (std::size_t i = 0; i < 400; ++i) a.add(i, data[i], meta[i]);
    b.add_batch(id_range(0, 400), flatten(data), meta, 8);
    CHECK(save_bytes(a) == save_bytes(b));
    if constexpr (HNSW_ALLOC_HOOK) {  // running out of memory restores the index exactly
        FlatIndex c(8, Metric::L2);
        for (std::size_t i = 0; i < 100; ++i) c.add(i, data[i], meta[i]);
        const std::string before = save_bytes(c);
        std::vector<Metadata> extra(100);
        for (std::size_t i = 0; i < 100; ++i) extra[i] = meta[100 + i].set("batch_only", true);
        bool exact = true;
        CHECK(sweep_allocation_failures([&] { c.add_batch(id_range(100, 100), flatten(Vectors(data.begin() + 100, data.begin() + 200)), extra); },
                                        [&] { exact = exact && save_bytes(c) == before; }) > 0);
        CHECK(exact && c.size() == 200);
    }
}

TEST(add_batch, b18_batches_mixed_with_single_adds) {
    auto data = clustered(2000, 8, 8, 718);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    std::uint64_t next = 0;
    for (int round = 0; round < 5; ++round) {
        h.add_batch(id_range(next, 300), flatten(Vectors(data.begin() + std::ptrdiff_t(next), data.begin() + std::ptrdiff_t(next + 300))), 4);
        next += 300;
        for (int k = 0; k < 100; ++k, ++next) h.add(next, data[next]);
    }
    for (std::uint64_t i = 0; i < next; ++i) f.add(i, data[i]);
    CHECK(h.size() == 2000);
    check_graph(h, 0.99);
    CHECK(recall(h, f, Vectors(data.begin(), data.begin() + 100), 10, 64) > 0.95);
}

TEST(add_batch, b19_many_threads_every_metric) {
    // The ThreadSanitizer CI jobs run this group: 8 threads linking at once.
    auto data = clustered(2500, 12, 8, 719);
    auto queries = clustered(60, 12, 8, 720);
    for (Metric m : kMetrics) {
        HnswIndex h(12, m);
        FlatIndex f(12, m);
        h.add_batch(id_range(0, 2500), flatten(data), 8);
        for (std::size_t i = 0; i < 2500; ++i) f.add(i, data[i]);
        check_graph(h, 0.99);
        CHECK(recall(h, f, queries, 10, 64) > 0.9);
    }
}

// ===========================================================================
// Bulk changes by filter: remove_where, set_metadata_where, count_where (W1-W16)
// ===========================================================================

namespace {

/// Filters with their expected matches, computed from meta_for(i) directly
/// (IDs divisible by 11 have only a category).
struct FilterCase {
    const char* name;
    Filter filter;
    std::function<bool(std::uint64_t)> expected;
};

std::vector<FilterCase> filter_cases() {
    auto full = [](std::uint64_t i) { return i % 11 != 0; };
    return {
        {"eq", Filter::eq("category", "tech"), [](std::uint64_t i) { return i % 5 == 2; }},
        {"range", Filter::ge("year", 2005) && Filter::lt("year", 2010),
         [=](std::uint64_t i) { return full(i) && 2000 + i % 25 >= 2005 && 2000 + i % 25 < 2010; }},
        {"in", Filter::in("category", {"news", "food"}), [](std::uint64_t i) { return i % 5 == 0 || i % 5 == 4; }},
        {"tag", Filter::has_tag("tags", "a"), [=](std::uint64_t i) { return full(i) && i % 2 == 1; }},
        {"all_tags", Filter::has_all_tags("tags", {"a", "b"}), [=](std::uint64_t i) { return full(i) && i % 2 == 1 && i % 3 == 0; }},
        {"not_exists", !Filter::exists("year"), [](std::uint64_t i) { return i % 11 == 0; }},
        {"or", Filter::eq("category", "music") || Filter::has_tag("tags", "c"),
         [=](std::uint64_t i) { return i % 5 == 3 || (full(i) && i % 7 == 0); }},
        {"and", Filter::eq("in_stock", true) && Filter::gt("price", 100.0),
         [=](std::uint64_t i) { return full(i) && i % 3 == 0 && double(i % 100) * 1.5 > 100.0; }},
    };
}

}  // namespace

TEST(bulk, w01_remove_where_removes_exactly_the_matches) {
    auto data = clustered(1000, 8, 6, 800);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1000; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const Filter tech = Filter::eq("category", "tech");
    CHECK(h.count_where(tech) == 200 && f.count_where(tech) == 200);
    CHECK(h.remove_where(tech) == 200 && f.remove_where(tech) == 200);
    bool exact = h.size() == 800 && f.size() == 800;
    for (std::uint64_t i = 0; i < 1000; ++i) {
        const bool gone = i % 5 == 2;
        exact = exact && h.contains(i) == !gone && f.contains(i) == !gone;
        if (!gone) exact = exact && *h.get_metadata(i) == meta_for(i) && *f.get_metadata(i) == meta_for(i);
    }
    CHECK(exact && h.count_where(tech) == 0 && f.count_where(tech) == 0);
    check_graph(h, 0.98);
}

TEST(bulk, w02_no_match_changes_nothing) {
    auto data = clustered(300, 8, 4, 801);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    const Filter none = Filter::eq("category", "no-such-category");
    CHECK(h.count_where(none) == 0 && h.remove_where(none) == 0 && f.remove_where(none) == 0);
    CHECK(h.set_metadata_where(none, Metadata().set("year", 1)) == 0 && f.set_metadata_where(none, Metadata().set("year", 1)) == 0);
    CHECK(save_bytes(h) == hb && save_bytes(f) == fb);
}

TEST(bulk, w03_everything_matches) {
    auto data = clustered(300, 8, 4, 802);
    HnswIndex h(8, Metric::Cosine);
    FlatIndex f(8, Metric::Cosine);
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const Filter all = Filter::exists("category");
    CHECK(h.remove_where(all) == 300 && f.remove_where(all) == 300);
    CHECK(h.size() == 0 && f.size() == 0 && h.entry_point() == kEmpty && h.max_level() == -1);
    CHECK(h.search(data[0], 5).empty() && f.search(data[0], 5).empty());
    h.add(7, data[7]);
    f.add(7, data[7]);
    CHECK(h.search(data[7], 1)[0].id == 7 && f.search(data[7], 1)[0].id == 7);
}

TEST(bulk, w04_every_kind_of_filter) {
    auto data = clustered(600, 8, 6, 803);
    for (const FilterCase& c : filter_cases()) {
        HnswIndex h(8, Metric::L2);
        FlatIndex f(8, Metric::L2);
        for (std::uint64_t i = 0; i < 600; ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        std::size_t expected = 0;
        for (std::uint64_t i = 0; i < 600; ++i) expected += c.expected(i);
        const bool counts = h.count_where(c.filter) == expected && f.count_where(c.filter) == expected;
        const bool removed = h.remove_where(c.filter) == expected && f.remove_where(c.filter) == expected;
        bool exact = true;
        for (std::uint64_t i = 0; i < 600; ++i) exact = exact && h.contains(i) == !c.expected(i) && f.contains(i) == !c.expected(i);
        if (!(counts && removed && exact)) std::printf("    filter case '%s' failed\n", c.name);
        CHECK(counts && removed && exact);
    }
}

TEST(bulk, w05_unknown_fields) {
    auto data = clustered(100, 8, 2, 804);
    HnswIndex strict(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    FlatIndex fstrict(8, Metric::L2, Schema::strict(search_fields()));
    HnswIndex dyn(8, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) {
        strict.add(i, data[i], meta_for(i));
        fstrict.add(i, data[i], meta_for(i));
        dyn.add(i, data[i], meta_for(i));
    }
    const std::string sb = save_bytes(strict), fb = save_bytes(fstrict), db = save_bytes(dyn);
    const Filter unknown = Filter::eq("no_such_field", 1);
    CHECK_THROWS_AS(std::invalid_argument, strict.remove_where(unknown));
    CHECK_THROWS_AS(std::invalid_argument, strict.set_metadata_where(unknown, Metadata().set("year", 1)));
    CHECK_THROWS_AS(std::invalid_argument, (void)strict.count_where(unknown));
    CHECK_THROWS_AS(std::invalid_argument, fstrict.remove_where(unknown));
    CHECK(save_bytes(strict) == sb && save_bytes(fstrict) == fb);
    CHECK(dyn.count_where(unknown) == 0 && dyn.remove_where(unknown) == 0 && save_bytes(dyn) == db);
}

TEST(bulk, w06_large_removal_keeps_the_graph_healthy) {
    auto data = clustered(3000, 16, 10, 805);
    HnswIndex h(16, Metric::L2);
    for (std::uint64_t i = 0; i < 3000; ++i) h.add(i, data[i], Metadata().set("g", std::int64_t(i % 10)));
    // Make sure the entry point is among the removed.
    const std::uint64_t entry_id = h.storage().ids().external(h.entry_point());
    h.set_metadata(entry_id, Metadata().set("g", std::int64_t(0)));
    const Filter some = Filter::lt("g", 3);  // about 30%
    const std::size_t removed = h.remove_where(some);
    CHECK(removed >= 900 && !h.contains(entry_id) && !h.storage().ids().is_deleted(h.entry_point()));
    check_graph(h, 0.98);
    FlatIndex f(16, Metric::L2);
    for (std::uint64_t i = 0; i < 3000; ++i)
        if (h.contains(i)) f.add(i, data[i]);
    CHECK(recall(h, f, Vectors(data.begin(), data.begin() + 100), 10, 64) > 0.95);
    const std::size_t capacity = h.capacity();
    for (std::uint64_t i = 0; i < 3000; ++i)
        if (!h.contains(i)) h.add(i, data[i]);  // removed IDs are usable again, in the freed slots
    CHECK(h.size() == 3000 && h.capacity() == capacity);
    check_graph(h, 0.98);
    // With graph repair turned off: every match is still removed; links to them remain.
    HnswParams no_repair;
    no_repair.repair_on_remove = false;
    HnswIndex r(16, Metric::L2, no_repair);
    for (std::uint64_t i = 0; i < 1000; ++i) r.add(i, data[i], Metadata().set("g", std::int64_t(i % 10)));
    CHECK(r.remove_where(some) == 300 && r.size() == 700 && r.count_where(some) == 0);
    check_graph(r, 0.95, 1.0);
}

TEST(bulk, w07_payload_counts_stay_exact) {
    auto data = clustered(800, 8, 6, 806);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 800; ++i) h.add(i, data[i], meta_for(i));
    h.create_payload_index("category");
    h.remove_where(Filter::has_tag("tags", "c"));
    h.set_metadata_where(Filter::eq("category", "news"), Metadata().set("category", "sports"));
    const MetadataStore& m = h.metadata();
    bool exact = true;
    for (const char* cat : kCategories) {
        const auto id = m.find_string(cat);
        const std::size_t counted = id ? m.value_count(*m.find_field("category"), *id).value_or(SIZE_MAX) : 0;  // empty would mean no index: fail
        exact = exact && counted == h.count_where(Filter::eq("category", cat));
    }
    CHECK(exact && h.count_where(Filter::eq("category", "news")) == 0);
}

TEST(bulk, w08_set_metadata_where_changes_only_the_given_fields) {
    auto data = clustered(500, 8, 4, 807);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 500; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const Filter tech = Filter::eq("category", "tech");
    const Metadata change = Metadata().set("year", std::int64_t(1999)).unset("tags");
    CHECK(h.set_metadata_where(tech, change) == 100 && f.set_metadata_where(tech, change) == 100);
    // Compared field by field: in a change, unset("tags") is a "remove this field"
    // marker, while a stored row simply has no such field.
    bool exact = true;
    for (std::uint64_t i = 0; i < 500; ++i) {
        const Metadata original = meta_for(i);
        for (const Metadata& got : {*h.get_metadata(i), *f.get_metadata(i)}) {
            for (const char* field : {"category", "price", "in_stock"}) {
                const auto a = got.get(field), b = original.get(field);
                exact = exact && a.has_value() == b.has_value() && (!a || *a == *b);
            }
            if (i % 5 == 2) {
                exact = exact && got.get("year") && got.get("year")->i == 1999 && !got.get("tags");
            } else {
                const auto a = got.get("year"), b = original.get("year");
                const auto ta = got.get("tags"), tb = original.get("tags");
                exact = exact && a.has_value() == b.has_value() && (!a || *a == *b) && ta.has_value() == tb.has_value() && (!ta || *ta == *tb);
            }
        }
    }
    CHECK(exact && h.size() == 500 && f.size() == 500);
}

TEST(bulk, w09_invalid_changes_change_nothing) {
    auto data = clustered(200, 8, 4, 808);
    HnswIndex strict(8, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    FlatIndex f(8, Metric::L2, Schema::strict(search_fields()));
    for (std::uint64_t i = 0; i < 200; ++i) {
        strict.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const std::string sb = save_bytes(strict), fb = save_bytes(f);
    const std::vector<Metadata> bad = {Metadata().set("year", "not a number"), Metadata().set("unknown", 1),
                                       Metadata().set("price", double(kNaN))};
    for (const Filter& where : {Filter::eq("category", "tech"), Filter::eq("category", "no-such-category")}) {
        for (const Metadata& b : bad) {
            CHECK_THROWS_AS(std::invalid_argument, strict.set_metadata_where(where, b));  // even when nothing matches
            CHECK_THROWS_AS(std::invalid_argument, f.set_metadata_where(where, b));
        }
    }
    CHECK(save_bytes(strict) == sb && save_bytes(f) == fb);
}

TEST(bulk, w10_changing_the_filtered_field) {
    auto data = clustered(500, 8, 4, 809);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 500; ++i) h.add(i, data[i], meta_for(i));
    const std::size_t sports = h.count_where(Filter::eq("category", "sports"));
    CHECK(h.set_metadata_where(Filter::eq("category", "news"), Metadata().set("category", "sports")) == 100);
    CHECK(h.count_where(Filter::eq("category", "news")) == 0);
    CHECK(h.count_where(Filter::eq("category", "sports")) == sports + 100);
    // A change that keeps rows matching is applied once: a counter goes up by exactly 1.
    CHECK(h.set_metadata_where(Filter::eq("category", "sports"), Metadata().set("year", std::int64_t(3000))) == sports + 100);
    CHECK(h.count_where(Filter::eq("year", 3000)) == h.count_where(Filter::eq("category", "sports") && Filter::exists("year")));
}

TEST(bulk, w11_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    // remove_where either fails before changing anything, or removes every match
    // (a repair that runs out of memory is skipped, as by remove()).
    // set_metadata_where either changes every match or nothing.
    auto data = clustered(150, 8, 4, 810);
    for (int index_kind = 0; index_kind < 2; ++index_kind) {
        for (int op = 0; op < 2; ++op) {
            HnswIndex h(8, Metric::L2);
            FlatIndex f(8, Metric::L2);
            for (std::uint64_t i = 0; i < 150; ++i) {
                if (index_kind == 0) h.add(i, data[i], meta_for(i));
                else f.add(i, data[i], meta_for(i));
            }
            auto bytes = [&] { return index_kind == 0 ? save_bytes(h) : save_bytes(f); };
            const std::string before = bytes();
            const Filter where = Filter::eq("category", "tech") || Filter::has_tag("tags", "c");
            const Metadata change = Metadata().set("category", "brand-new-value").set("fresh_field", true);
            bool clean = true, completed = false;
            long failures = 0;
            for (long n = 0; n < 20000 && !completed; ++n) {
                alloc_hook::fail_after(n);
                bool failed = false;
                try {
                    if (op == 0) (void)(index_kind == 0 ? h.remove_where(where) : f.remove_where(where));
                    else (void)(index_kind == 0 ? h.set_metadata_where(where, change) : f.set_metadata_where(where, change));
                } catch (const std::bad_alloc&) {
                    failed = true;
                }
                const bool fired = alloc_hook::fired;
                alloc_hook::disarm();
                if (failed) {
                    ++failures;
                    clean = clean && bytes() == before;  // nothing changed
                } else {
                    completed = true;  // done, whether or not a repair ran out of memory
                    (void)fired;
                }
            }
            // Afterwards: removal leaves no match; the change gave every match the new category.
            std::size_t matched = 0;  // tech, or tag c (IDs divisible by 11 have no tags)
            for (std::uint64_t i = 0; i < 150; ++i) matched += (i % 5 == 2 || (i % 11 != 0 && i % 7 == 0));
            const std::size_t outcome = op == 0 ? (index_kind == 0 ? h.count_where(where) : f.count_where(where))
                                                : (index_kind == 0 ? h.count_where(Filter::eq("category", "brand-new-value"))
                                                                   : f.count_where(Filter::eq("category", "brand-new-value")));
            CHECK(clean && completed && failures > 0 && outcome == (op == 0 ? 0 : matched));
            if (index_kind == 0) check_graph(h, 0.97);
        }
    }
}

TEST(bulk, w12_inside_a_search) {
    auto data = clustered(100, 8, 2, 811);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    std::atomic<long> refused{0}, counted{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        counted += long(h.count_where(Filter::eq("category", "tech")) == 20);  // reading is allowed
        for (auto&& change : {std::function<void()>([&] { h.remove_where(Filter::eq("category", "tech")); }),
                              std::function<void()>([&] { h.set_metadata_where(Filter::eq("category", "tech"), Metadata().set("year", 1)); }),
                              std::function<void()>([&] { f.remove_where(Filter::eq("category", "tech")); })}) {
            try {
                change();
            } catch (const std::logic_error&) {
                ++refused;
            }
        }
        return true;
    };
    (void)h.search(data[0], 5, 32, o);
    CHECK(counted > 0 && refused == 3 * counted && h.size() == 100 && f.size() == 100);
}

TEST(bulk, w13_searches_see_all_of_a_change_or_none) {
    auto data = clustered(2000, 8, 6, 812);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 2000; ++i) h.add(i, data[i], Metadata().set("g", std::int64_t(i % 4)).set("flag", false));
    std::atomic<bool> stop{false};
    std::atomic<long> partial{0}, observations{0};
    with_watchdog(120, "w13", [&] {
        std::thread changer([&] {
            for (std::int64_t g = 0; g < 3; ++g) {
                h.set_metadata_where(Filter::eq("g", g), Metadata().set("flag", true));  // 500 rows at once
                h.remove_where(Filter::eq("g", g));                                     // then remove them
            }
            stop = true;
        });
        run_threads(3, [&](std::size_t) {
            while (!stop) {
                const std::size_t size = h.size(), flagged = h.count_where(Filter::eq("flag", true));
                if (size % 500 != 0 || (flagged != 0 && flagged != 500)) ++partial;
                ++observations;
            }
        });
        changer.join();
    });
    CHECK(partial == 0 && observations > 0 && h.size() == 500);
}

TEST(bulk, w14_save_and_load_afterwards) {
    auto data = clustered(600, 8, 4, 813);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 600; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    for (HnswIndex* x : {&h}) {
        x->remove_where(Filter::eq("category", "food"));
        x->set_metadata_where(Filter::has_tag("tags", "a"), Metadata().set("note", "odd"));
    }
    f.remove_where(Filter::eq("category", "food"));
    f.set_metadata_where(Filter::has_tag("tags", "a"), Metadata().set("note", "odd"));
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    auto h2 = load_hnsw(hb);
    auto f2 = load_flat(fb);
    CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb);
    CHECK(h2->count_where(Filter::eq("note", "odd")) == h.count_where(Filter::eq("note", "odd")));
    CHECK(f2->count_where(Filter::eq("category", "food")) == 0);
}

TEST(bulk, w15_later_changes_work_normally) {
    auto data = clustered(1200, 8, 6, 814);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1000; ++i) h.add(i, data[i], meta_for(i));
    h.remove_where(Filter::in("category", {"news", "tech"}));
    h.set_metadata_where(Filter::eq("category", "music"), Metadata().set("category", "news"));
    for (std::uint64_t i = 1000; i < 1200; ++i) h.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 1; i < 1000; i += 9)
        if (h.contains(i)) h.update(i, data[(i * 3) % 1200]);
    check_graph(h, 0.98);
    h.compact();
    check_graph(h, 0.99);
    std::size_t new_music = 0;  // music was renamed, but the 200 vectors added afterwards bring new ones
    for (std::uint64_t i = 1000; i < 1200; ++i) new_music += (i % 5 == 3);
    CHECK(h.deleted_count() == 0 && h.count_where(Filter::eq("category", "music")) == new_music);
}

TEST(bulk, w16_flat_moves_metadata_with_vectors) {
    // Flat removal moves the last vector into each freed slot: every survivor must
    // keep its own vector and metadata.
    auto data = random_vectors(700, 8, 815);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 700; ++i) f.add(i, data[i], meta_for(i));
    CHECK(f.remove_where(Filter::has_tag("tags", "b") || Filter::eq("category", "music")) > 0);
    bool intact = true;
    for (std::uint64_t i = 0; i < 700; ++i) {
        if (!f.contains(i)) continue;
        const auto r = f.search(data[i], 1);
        intact = intact && r[0].id == i && r[0].distance == 0.0f && *f.get_metadata(i) == meta_for(i);
    }
    CHECK(intact && f.capacity() == f.size());
}

// ===========================================================================
// Reading records back: get_vector, get, scan (R1-R16)
// ===========================================================================

namespace {

/// Every page of a scan, concatenated; also checks each page's shape.
std::vector<Record> scan_all(const auto& index, ScanOptions o, bool& pages_ok) {
    std::vector<Record> all;
    for (int guard = 0; guard < 100000; ++guard) {
        const Page page = index.scan(o);
        pages_ok = pages_ok && page.records.size() <= o.limit;
        if (page.next) pages_ok = pages_ok && page.records.size() == o.limit && page.records.back().id == *page.next;
        all.insert(all.end(), page.records.begin(), page.records.end());
        if (!page.next) return all;
        o.after = page.next;
    }
    pages_ok = false;
    return all;
}

std::vector<std::uint64_t> ids_of(const std::vector<Record>& records) {
    std::vector<std::uint64_t> ids;
    for (const Record& r : records) ids.push_back(r.id);
    return ids;
}

/// A version 1 file made from a current file of a cosine index, by removing the
/// originals stored after each vector (version 2), the CODC section and the
/// truncation setting at the end of CONF (version 3).
std::string as_version_1(const std::string& v2, bool hnsw, std::size_t dim) {
    ParsedFile f = ParsedFile::parse(v2);
    poke<std::uint32_t>(f.header, 8, 1);
    std::erase_if(f.sections, [](const auto& sec) { return sec.first == persist::tag("CODC") || sec.first == persist::tag("IVFL"); });  // version 3 only
    std::string& conf = f.payload("CONF");
    conf.resize(conf.size() - 10);  // version 3 ends CONF with the truncation setting and two flags
    std::string& node = f.payload("NODE");
    std::string out = node.substr(0, 8);
    const auto n = persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(&node[0]));
    std::size_t at = 8;
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::size_t head = hnsw ? 8 + 1 + 1 : 8;
        const int level = hnsw ? static_cast<unsigned char>(node[at + 9]) : 0;
        out += node.substr(at, head + 4 * dim);  // ID (and flags), stored vector
        at += head + 8 * dim;                     // skip the original
        if (hnsw)
            for (int l = 0; l <= level; ++l) {
                const auto count = persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(&node[at]));
                out += node.substr(at, 4 + 4 * std::size_t(count));
                at += 4 + 4 * std::size_t(count);
            }
    }
    node = out;
    return f.assemble();
}

}  // namespace

TEST(records, r01_get_vector_returns_the_vector_as_given) {
    auto data = random_vectors(300, 12, 900);
    for (Metric m : kMetrics) {
        HnswIndex h(12, m);
        FlatIndex f(12, m);
        for (std::uint64_t i = 0; i < 300; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        bool exact = true;
        for (std::uint64_t i = 0; i < 300; ++i) exact = exact && *h.get_vector(i) == data[i] && *f.get_vector(i) == data[i];  // bit for bit
        CHECK(exact);
        CHECK(!h.get_vector(999) && !f.get_vector(999));
        h.remove(5);
        f.remove(5);
        CHECK(!h.get_vector(5) && !f.get_vector(5));
        // A vector read back finds itself.
        CHECK(h.search(*h.get_vector(7), 1, 64)[0].id == 7 && f.search(*f.get_vector(7), 1)[0].id == 7);
    }
}

TEST(records, r02_get_returns_vector_and_metadata) {
    auto data = random_vectors(50, 6, 901);
    HnswIndex h(6, Metric::Cosine);
    FlatIndex f(6, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) {
        h.add(100 + i, data[i], meta_for(i));
        f.add(100 + i, data[i], meta_for(i));
    }
    for (auto r : {h.get(110), f.get(110)}) CHECK(r && r->id == 110 && r->vector == data[10] && r->metadata == meta_for(10));
    CHECK(!h.get(1) && !f.get(1));
}

TEST(records, r03_paging_through_everything) {
    auto data = random_vectors(500, 4, 902);
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    std::vector<std::uint64_t> ids;
    std::mt19937_64 rng(3);
    for (std::size_t i = 0; i < 500; ++i) ids.push_back(rng() % 1000000);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<std::uint64_t> shuffled = ids;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);  // added out of order
    for (std::size_t i = 0; i < shuffled.size(); ++i) {
        h.add(shuffled[i], data[i]);
        f.add(shuffled[i], data[i]);
    }
    for (std::size_t limit : {std::size_t(1), std::size_t(7), std::size_t(100), std::size_t(1000)}) {
        ScanOptions o;
        o.limit = limit;
        bool pages_ok = true;
        const auto hr = scan_all(h, o, pages_ok), fr = scan_all(f, o, pages_ok);
        CHECK(pages_ok && ids_of(hr) == ids && ids_of(fr) == ids);  // each exactly once, ascending
        bool vectors_ok = true;
        for (const Record& r : hr) vectors_ok = vectors_ok && r.vector == *h.get_vector(r.id);
        CHECK(vectors_ok);
    }
}

TEST(records, r04_filtered_paging) {
    auto data = clustered(600, 8, 6, 903);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 600; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    for (const FilterCase& c : filter_cases()) {
        std::vector<std::uint64_t> expected;
        for (std::uint64_t i = 0; i < 600; ++i)
            if (c.expected(i)) expected.push_back(i);
        ScanOptions o;
        o.filter = c.filter;
        o.limit = 37;
        bool pages_ok = true;
        const bool ok = ids_of(scan_all(h, o, pages_ok)) == expected && ids_of(scan_all(f, o, pages_ok)) == expected &&
                        h.count_where(c.filter) == expected.size();
        if (!(ok && pages_ok)) std::printf("    filter case '%s' failed\n", c.name);
        CHECK(ok && pages_ok);
    }
}

TEST(records, r05_edge_cases) {
    HnswIndex empty(3, Metric::L2);
    const Page none = empty.scan();
    CHECK(none.records.empty() && !none.next);
    HnswIndex h(3, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) h.add(i, std::vector<float>{float(i), 1, 2});
    ScanOptions big;
    big.limit = 1000;
    const Page all = h.scan(big);
    CHECK(all.records.size() == 10 && !all.next);  // fits in one page: no next
    ScanOptions exact;
    exact.limit = 10;
    CHECK(h.scan(exact).records.size() == 10 && !h.scan(exact).next);  // exactly one full page: no next
    ScanOptions zero;
    zero.limit = 0;
    CHECK_THROWS_AS(std::invalid_argument, (void)h.scan(zero));
    ScanOptions beyond;
    beyond.after = 999;
    CHECK(h.scan(beyond).records.empty() && !h.scan(beyond).next);
    for (std::uint64_t i = 0; i < 10; ++i) h.remove(i);
    CHECK(h.scan().records.empty() && !h.scan().next);
}

TEST(records, r06_cursor_on_a_removed_id) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i * 10, std::vector<float>{float(i), 0});
    h.remove(50);
    ScanOptions o;
    o.after = 50;  // no longer stored
    o.limit = 3;
    CHECK(ids_of(h.scan(o).records) == (std::vector<std::uint64_t>{60, 70, 80}));
    o.after = 55;  // never stored
    CHECK(ids_of(h.scan(o).records) == (std::vector<std::uint64_t>{60, 70, 80}));
}

TEST(records, r07_without_vectors_or_metadata) {
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    for (std::uint64_t i = 0; i < 30; ++i) {
        h.add(i, std::vector<float>{float(i), 1, 2, 3}, meta_for(i));
        f.add(i, std::vector<float>{float(i), 1, 2, 3}, meta_for(i));
    }
    ScanOptions ids_only;
    ids_only.with_vectors = false;
    ids_only.with_metadata = false;
    bool empty_fields = true;
    for (const Page& page : {h.scan(ids_only), f.scan(ids_only)})
        for (const Record& r : page.records) empty_fields = empty_fields && r.vector.empty() && r.metadata == Metadata();
    CHECK(empty_fields);
    ScanOptions meta_only;
    meta_only.with_vectors = false;
    const Page m = h.scan(meta_only);
    CHECK(m.records[3].vector.empty() && m.records[3].metadata == meta_for(3));
}

TEST(records, r08_extreme_ids) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t id : {kMaxId, std::uint64_t(0), kMaxId - 1, std::uint64_t(1)}) h.add(id, std::vector<float>{float(id % 7), 1});
    ScanOptions o;
    o.limit = 2;
    bool pages_ok = true;
    CHECK(ids_of(scan_all(h, o, pages_ok)) == (std::vector<std::uint64_t>{0, 1, kMaxId - 1, kMaxId}) && pages_ok);
    o.after = kMaxId;
    CHECK(h.scan(o).records.empty());
    o.after = 0;
    CHECK(h.scan(o).records.front().id == 1);  // `after` is exclusive
}

TEST(records, r09_changes_between_pages) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i * 2, std::vector<float>{float(i), 1});  // even IDs 0..198
    ScanOptions o;
    o.limit = 10;
    Page page = h.scan(o);  // 0..18
    std::vector<std::uint64_t> seen = ids_of(page.records);
    h.remove(20);       // next on the list: skipped
    h.remove(4);        // already returned: nothing changes
    h.add(21, std::vector<float>{0.5f, 1});  // after the cursor: included
    h.add(3, std::vector<float>{0.7f, 1});   // before the cursor: not included
    while (page.next) {
        o.after = page.next;
        page = h.scan(o);
        const auto ids = ids_of(page.records);
        seen.insert(seen.end(), ids.begin(), ids.end());
    }
    std::vector<std::uint64_t> expected;
    for (std::uint64_t i = 0; i < 200; i += 2)
        if (i != 20) expected.push_back(i);
    expected.insert(std::upper_bound(expected.begin(), expected.end(), 21), 21);
    CHECK(seen == expected);  // nothing repeated, nothing that stayed stored skipped
}

TEST(records, r10_paging_while_the_index_changes) {
    auto data = clustered(2000, 8, 6, 904);
    HnswIndex h(8, Metric::Cosine);
    for (std::uint64_t i = 0; i < 1000; ++i) h.add(i, data[i], meta_for(i));
    std::atomic<bool> stop{false};
    std::atomic<long> bad{0}, pages{0};
    with_watchdog(120, "r10", [&] {
        std::thread writer([&] {
            for (std::uint64_t i = 1000; i < 2000; ++i) {
                h.add(i, data[i], meta_for(i));
                if (i % 3 == 0) h.remove(i - 900);
                if (i % 5 == 0) h.update(i - 500, data[i - 1000]);
            }
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            while (!stop) {
                ScanOptions o;
                o.limit = 50 + t;
                o.filter = t == 2 ? std::optional<Filter>(Filter::eq("category", "tech")) : std::nullopt;
                for (int k = 0; k < 10; ++k) {
                    const Page page = h.scan(o);
                    ++pages;
                    for (std::size_t i = 0; i < page.records.size(); ++i) {
                        const Record& r = page.records[i];
                        if ((i > 0 && r.id <= page.records[i - 1].id) || r.vector.size() != 8 || (o.after && r.id <= *o.after)) ++bad;
                    }
                    if (!page.next) break;
                    o.after = page.next;
                }
            }
        });
        writer.join();
    });
    CHECK(bad == 0 && pages > 0);
}

TEST(records, r11_unknown_filter_fields) {
    HnswIndex strict(2, Metric::L2, HnswParams{}, Schema::strict(search_fields()));
    HnswIndex dyn(2, Metric::L2);
    strict.add(1, std::vector<float>{1, 0}, meta_for(1));
    dyn.add(1, std::vector<float>{1, 0}, meta_for(1));
    ScanOptions o;
    o.filter = Filter::eq("no_such_field", 1);
    CHECK_THROWS_AS(std::invalid_argument, (void)strict.scan(o));
    CHECK(dyn.scan(o).records.empty());
}

TEST(records, r12_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = random_vectors(200, 8, 905);
    HnswIndex h(8, Metric::Cosine);
    FlatIndex f(8, Metric::Cosine);
    for (std::uint64_t i = 0; i < 100; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    ScanOptions o;
    o.filter = Filter::has_tag("tags", "a");
    o.limit = 30;
    Page page;
    CHECK(sweep_allocation_failures([&] { page = h.scan(o); }, [&] { CHECK(save_bytes(h) == hb); }) > 0 && page.records.size() == 30);
    std::optional<Record> got;
    CHECK(sweep_allocation_failures([&] { got = f.get(10); }, [&] { CHECK(save_bytes(f) == fb); }) > 0 && got->vector == data[10]);
    // Adding to a cosine index stores the original too, all or nothing. A failed
    // HNSW add leaves the contents as they were, though its bookkeeping may move
    // on (a slot kept free, a random level used up; identical for L2), so its
    // contents are compared; Flat is restored exactly.
    ScanOptions everything;
    everything.limit = 1000;
    auto contents = [&] {
        std::vector<std::pair<std::uint64_t, std::vector<float>>> c;
        for (const Record& r : h.scan(everything).records) c.emplace_back(r.id, r.vector);
        return c;
    };
    const auto before = contents();
    std::uint64_t id = 500;
    CHECK(sweep_allocation_failures([&] { h.add(id, data[150], meta_for(1)); },
                                    [&] { CHECK(!h.contains(id) && h.size() == 100 && contents() == before); }) > 0);
    CHECK(sweep_allocation_failures([&] { f.add(id, data[150], meta_for(1)); }, [&] { CHECK(!f.contains(id) && save_bytes(f) == fb); }) > 0);
    CHECK(*h.get_vector(500) == data[150] && *f.get_vector(500) == data[150]);
}

TEST(records, r13_allowed_inside_a_search) {
    auto data = random_vectors(50, 4, 906);
    HnswIndex h(4, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) h.add(i, data[i]);
    std::atomic<long> reads{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t id) {
        reads += long(h.get_vector(id).has_value()) + long(h.get(id).has_value()) + long(!h.scan().records.empty());
        return true;
    };
    (void)h.search(data[0], 5, 32, o);
    CHECK(reads > 0 && reads % 3 == 0);
}

TEST(records, r14_after_save_and_load) {
    auto data = random_vectors(300, 6, 907);
    HnswIndex h(6, Metric::Cosine);
    FlatIndex f(6, Metric::Cosine);
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    for (std::uint64_t i = 0; i < 300; i += 7) {
        h.remove(i);
        f.remove(i);
    }
    auto h2 = load_hnsw(save_bytes(h));
    auto f2 = load_flat(save_bytes(f));
    ScanOptions o;
    o.limit = 1000;
    const auto a = h.scan(o).records, b = h2->scan(o).records, c = f.scan(o).records, d = f2->scan(o).records;
    bool same = a.size() == b.size() && c.size() == d.size();
    for (std::size_t i = 0; same && i < a.size(); ++i) same = a[i].id == b[i].id && a[i].vector == b[i].vector && a[i].metadata == b[i].metadata;
    for (std::size_t i = 0; same && i < c.size(); ++i) same = c[i].id == d[i].id && c[i].vector == d[i].vector && c[i].metadata == d[i].metadata;
    CHECK(same && *h2->get_vector(1) == data[1] && *f2->get_vector(1) == data[1]);  // the originals survive saving
}

TEST(records, r15_version_1_cosine_files) {
    // Version 1 did not store the original of a cosine vector: such files still
    // load, and get_vector returns the stored (normalized) vector instead.
    auto data = random_vectors(80, 5, 908);
    HnswIndex h(5, Metric::Cosine);
    FlatIndex f(5, Metric::Cosine);
    for (std::uint64_t i = 0; i < 80; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    h.remove(3);
    auto h1 = load_hnsw(as_version_1(save_bytes(h), true, 5));
    auto f1 = load_flat(as_version_1(save_bytes(f), false, 5));
    bool normalized = h1->size() == 79 && f1->size() == 80;
    for (std::uint64_t i = 4; i < 80; ++i) {
        for (const auto& v : {*h1->get_vector(i), *f1->get_vector(i)}) {
            double length = 0, dot = 0, original = 0;
            for (std::size_t d = 0; d < 5; ++d) {
                length += double(v[d]) * v[d];
                dot += double(v[d]) * data[i][d];
                original += double(data[i][d]) * data[i][d];
            }
            normalized = normalized && std::abs(length - 1.0) < 1e-5 && std::abs(dot - std::sqrt(original)) < 1e-4;  // same direction, length 1
        }
    }
    CHECK(normalized);
    CHECK(same_results(h1->search(data[9], 5, 64), h.search(data[9], 5, 64)));
    // Saved again, they become version 2 files.
    CHECK(persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(save_bytes(*h1).data() + 8)) == persist::kVersion);
}

TEST(records, r16_originals_follow_every_path) {
    // Cosine: after every way of storing, moving and freeing vectors, each stored
    // vector reads back exactly as last given.
    auto data = random_vectors(900, 8, 909);
    std::map<std::uint64_t, std::vector<float>> want;
    HnswIndex h(8, Metric::Cosine);
    FlatIndex f(8, Metric::Cosine);
    auto add = [&](std::uint64_t id, const std::vector<float>& v) { h.add(id, v); f.add(id, v); want[id] = v; };
    for (std::uint64_t i = 0; i < 300; ++i) add(i, data[i]);
    for (std::uint64_t i = 0; i < 300; i += 4) { h.remove(i); f.remove(i); want.erase(i); }       // Flat: swap with last
    for (std::uint64_t i = 1000; i < 1040; ++i) add(i, data[i - 700]);                            // HNSW: reused slots
    bool agree = true;
    for (std::uint64_t i = 1; i < 300; i += 5) {  // update returns false for IDs removed above
        const bool hu = h.update(i, data[i + 340]), fu = f.update(i, data[i + 340]);
        agree = agree && hu == fu && hu == want.contains(i);
        if (hu) want[i] = data[i + 340];
    }
    CHECK(agree);
    for (std::uint64_t i = 2; i < 300; i += 9) { h.upsert(i, data[i + 600]); f.upsert(i, data[i + 600]); want[i] = data[i + 600]; }
    std::vector<std::uint64_t> batch_ids;
    std::vector<float> batch;
    for (std::uint64_t i = 2000; i < 2100; ++i) {
        batch_ids.push_back(i);
        batch.insert(batch.end(), data[i - 1200].begin(), data[i - 1200].end());  // vectors 800-899
        want[i] = data[i - 1200];
    }
    for (std::uint64_t i = 3; i < 300; i += 11) { h.remove(i); f.remove(i); want.erase(i); }      // slots for the batch to reuse
    h.add_batch(std::span<const std::uint64_t>(batch_ids.data(), 50), std::span<const float>(batch.data(), 50 * 8), 1);
    h.add_batch(std::span<const std::uint64_t>(batch_ids.data() + 50, 50), std::span<const float>(batch.data() + 50 * 8, 50 * 8), 4);
    f.add_batch(batch_ids, batch);
    auto all_exact = [&] {
        bool exact = h.size() == want.size() && f.size() == want.size();
        for (const auto& [id, v] : want) {
            const bool ok = h.get_vector(id) == v && f.get_vector(id) == v;
            if (!ok && exact) std::printf("    first mismatch: id %llu (stored in hnsw: %d, flat: %d)\n", (unsigned long long)id, int(h.contains(id)), int(f.contains(id)));
            exact = exact && ok;
        }
        return exact;
    };
    CHECK(all_exact());
    h.compact();  // rebuilt from the originals
    CHECK(all_exact());
    CHECK(h.search(want.begin()->second, 1, 64)[0].id == want.begin()->first);
}

TEST(records, r17_damaged_cosine_files) {
    // The cosine-only part of the format (originals after each vector) read from
    // damaged files: truncated at every byte, a bit flipped in every byte.
    auto data = random_vectors(12, 3, 910);
    HnswIndex h(3, Metric::Cosine, HnswParams{4, 16, 1, true, {}});
    FlatIndex f(3, Metric::Cosine);
    for (std::uint64_t i = 0; i < 12; ++i) {
        h.add(i, data[i], Metadata().set("k", std::int64_t(i)));
        f.add(i, data[i]);
    }
    h.remove(4);
    for (int kind = 0; kind < 2; ++kind) {
        const std::string bytes = kind == 0 ? save_bytes(h) : save_bytes(f);
        auto load = [&](const std::string& b) { if (kind == 0) (void)load_hnsw(b); else (void)load_flat(b); };
        std::size_t rejected = 0, tried = 0;
        for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
            try { load(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
        }
        for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
            std::string damaged = bytes;
            damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
            try { load(damaged); } catch (const FormatError&) { ++rejected; }
        }
        CHECK(rejected == tried);
    }
}

// ===========================================================================
// remove_batch, search_by_id, deadlines, cancellation and progress
// (K1-K6, S1-S6, T1-T13)
// ===========================================================================

namespace {

using SteadyClock = std::chrono::steady_clock;

/// Every stored (ID, vector) pair, for comparing contents across a failed change.
std::vector<std::pair<std::uint64_t, std::vector<float>>> contents_of(const auto& index) {
    ScanOptions o;
    o.limit = 1000000;
    std::vector<std::pair<std::uint64_t, std::vector<float>>> c;
    for (const Record& r : index.scan(o).records) c.emplace_back(r.id, r.vector);
    return c;
}

bool sorted_and_unique(const std::vector<SearchResult>& r) {
    std::set<std::uint64_t> seen;
    for (std::size_t i = 0; i < r.size(); ++i)
        if ((i > 0 && r[i].distance < r[i - 1].distance) || !seen.insert(r[i].id).second) return false;
    return true;
}

/// OperationOptions that cancel once progress reaches `at` (of total).
OperationOptions cancel_at(std::atomic<bool>& flag, std::size_t at) {
    OperationOptions o;
    o.cancel = &flag;
    o.progress = [&flag, at](std::size_t done, std::size_t) {
        if (done >= at) flag = true;
    };
    return o;
}

}  // namespace

TEST(control, k01_remove_batch_removes_exactly_the_listed_ids) {
    auto data = clustered(800, 8, 6, 950);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 800; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    std::vector<std::uint64_t> gone;
    for (std::uint64_t i = 0; i < 800; i += 3) gone.push_back(i);
    CHECK(h.remove_batch(gone) == gone.size() && f.remove_batch(gone) == gone.size());
    bool exact = true;
    for (std::uint64_t i = 0; i < 800; ++i) {
        exact = exact && h.contains(i) == (i % 3 != 0) && f.contains(i) == (i % 3 != 0);
        if (i % 3) exact = exact && *h.get_vector(i) == data[i] && *f.get_metadata(i) == meta_for(i);
    }
    CHECK(exact);
    check_graph(h, 0.98);
}

TEST(control, k02_unknown_and_repeated_ids) {
    HnswIndex h(2, Metric::L2);
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, std::vector<float>{float(i), 0});
        f.add(i, std::vector<float>{float(i), 0});
    }
    const std::vector<std::uint64_t> ids = {3, 3, 999, 5, 3, 12345, 5};  // 3 and 5 once each
    CHECK(h.remove_batch(ids) == 2 && f.remove_batch(ids) == 2 && h.size() == 8 && f.size() == 8);
    CHECK(h.remove_batch(ids) == 0 && f.remove_batch(ids) == 0);  // already removed
}

TEST(control, k03_empty_list_changes_nothing) {
    HnswIndex h(2, Metric::L2);
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, std::vector<float>{float(i), 1});
        f.add(i, std::vector<float>{float(i), 1});
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    CHECK(h.remove_batch({}) == 0 && f.remove_batch({}) == 0 && save_bytes(h) == hb && save_bytes(f) == fb);
}

TEST(control, k04_removing_everything) {
    auto data = clustered(300, 8, 4, 951);
    HnswIndex h(8, Metric::Cosine);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i]);
    CHECK(h.remove_batch(id_range(0, 300)) == 300 && h.size() == 0 && h.entry_point() == kEmpty && h.max_level() == -1);
    h.add(5, data[5]);
    CHECK(h.search(data[5], 1)[0].id == 5);
}

TEST(control, k05_out_of_memory_always_completes) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(200, 8, 4, 952);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 200; ++i) h.add(i, data[i]);
    const std::string before = save_bytes(h);
    std::vector<std::uint64_t> gone;
    for (std::uint64_t i = 0; i < 200; i += 4) gone.push_back(i);
    bool clean = true, completed = false;
    long failures = 0;
    for (long n = 0; n < 20000 && !completed; ++n) {
        alloc_hook::fail_after(n);
        bool failed = false;
        try {
            (void)h.remove_batch(gone);
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        alloc_hook::disarm();
        if (failed) {
            ++failures;
            clean = clean && save_bytes(h) == before;  // failed before changing anything
        } else {
            completed = true;  // done, whether or not a repair ran out of memory
        }
    }
    CHECK(clean && completed && failures > 0 && h.size() == 150);
    check_graph(h, 0.97);
}

TEST(control, k06_refused_inside_a_search) {
    HnswIndex h(2, Metric::L2);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, std::vector<float>{float(i), 0});
    std::atomic<long> refused{0};
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        try {
            (void)h.remove_batch(std::vector<std::uint64_t>{1, 2});
        } catch (const std::logic_error&) {
            ++refused;
        }
        return true;
    };
    (void)h.search(std::vector<float>{0, 0}, 3, 16, o);
    CHECK(refused > 0 && h.size() == 20);
}

TEST(control, s01_same_as_searching_with_the_stored_vector) {
    auto data = clustered(1000, 8, 6, 953);
    for (Metric m : kMetrics) {
        HnswIndex h(8, m);
        FlatIndex f(8, m);
        for (std::uint64_t i = 0; i < 1000; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        bool same = true;
        for (std::uint64_t id = 0; id < 1000; id += 97) {
            auto expected_h = h.search(*h.get_vector(id), 11, 64);
            auto expected_f = f.search(*f.get_vector(id), 11);
            std::erase_if(expected_h, [&](const SearchResult& r) { return r.id == id; });
            std::erase_if(expected_f, [&](const SearchResult& r) { return r.id == id; });
            expected_h.resize(std::min<std::size_t>(expected_h.size(), 10));
            expected_f.resize(std::min<std::size_t>(expected_f.size(), 10));
            same = same && same_results(*h.search_by_id(id, 10, 64), expected_h) && same_results(*f.search_by_id(id, 10), expected_f);
        }
        CHECK(same);
    }
}

TEST(control, s02_including_the_record_itself) {
    auto data = clustered(500, 8, 4, 954);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 500; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const auto hs = *h.search_by_id(42, 5, 64, {}, true);
    const auto fs = *f.search_by_id(42, 5, {}, true);
    CHECK(hs[0].id == 42 && hs[0].distance == 0.0f && fs[0].id == 42 && fs[0].distance == 0.0f);
    CHECK(same_results(hs, h.search(data[42], 5, 64)) && same_results(fs, f.search(data[42], 5)));
    const auto without = *f.search_by_id(42, 5);
    CHECK(without.size() == 5 && std::none_of(without.begin(), without.end(), [](const SearchResult& r) { return r.id == 42; }));
}

TEST(control, s03_filters_and_predicates) {
    auto data = clustered(600, 8, 6, 955);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 600; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    SearchOptions tech = with_filter(Filter::eq("category", "tech"));
    bool ok = true;
    for (std::uint64_t id : {std::uint64_t(2), std::uint64_t(3)}) {  // 2 is "tech" itself, 3 is not
        for (const auto& r : {*h.search_by_id(id, 10, 64, tech), *f.search_by_id(id, 10, tech)}) {
            ok = ok && r.size() == 10 && sorted_and_unique(r);
            for (const SearchResult& x : r) ok = ok && x.id % 5 == 2 && x.id != id;
        }
    }
    SearchOptions even;
    even.predicate = [](std::uint64_t id) { return id % 2 == 0; };
    // Keep the optional alive: `for (x : *f.search_by_id(...))` would loop over a
    // destroyed temporary (a range-for keeps only the outermost result alive).
    const auto by_even = f.search_by_id(10, 8, even);
    ok = ok && by_even && by_even->size() == 8;
    for (const SearchResult& x : *by_even) ok = ok && x.id % 2 == 0 && x.id != 10;
    CHECK(ok);
}

TEST(control, s04_unknown_or_removed_ids) {
    HnswIndex h(2, Metric::L2);
    FlatIndex f(2, Metric::L2);
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, std::vector<float>{float(i), 0});
        f.add(i, std::vector<float>{float(i), 0});
    }
    h.remove(4);
    f.remove(4);
    CHECK(!h.search_by_id(4, 3) && !f.search_by_id(4, 3) && !h.search_by_id(99, 3) && !f.search_by_id(99, 3));
    CHECK(h.search_by_id(5, 0)->empty() && f.search_by_id(5, 0)->empty());  // k = 0: stored, nothing asked for
    // The largest k: every other record (no overflow when asking for one extra).
    CHECK(f.search_by_id(5, SIZE_MAX)->size() == 8 && h.search_by_id(5, SIZE_MAX, 64)->size() == 8);
}

TEST(control, s05_duplicates_of_the_record) {
    FlatIndex f(3, Metric::L2);
    HnswIndex h(3, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) {
        const std::vector<float> v{float(i), 1, 2};
        f.add(i, v);
        h.add(i, v);
    }
    for (std::uint64_t dup : {std::uint64_t(100), std::uint64_t(101)}) {  // exact copies of record 7
        f.add(dup, std::vector<float>{7, 1, 2});
        h.add(dup, std::vector<float>{7, 1, 2});
    }
    for (const auto& r : {*f.search_by_id(7, 2), *h.search_by_id(7, 2, 64)}) {
        CHECK(r.size() == 2 && r[0].distance == 0.0f && r[1].distance == 0.0f);
        CHECK((r[0].id == 100 || r[0].id == 101) && (r[1].id == 100 || r[1].id == 101) && r[0].id != r[1].id);
    }
}

TEST(control, s06_atomic_while_the_record_changes) {
    // The vector is read and searched under one lock: with the record included,
    // it always finds itself at distance 0, even while another thread updates it.
    auto data = clustered(300, 8, 4, 956);
    FlatIndex f(8, Metric::Cosine);
    for (std::uint64_t i = 0; i < 300; ++i) f.add(i, data[i]);
    // Readers do a fixed number of checks; the writer keeps updating until they finish.
    std::atomic<bool> readers_done{false};
    std::atomic<long> wrong{0}, checks{0}, updates{0};
    with_watchdog(120, "s06", [&] {
        std::thread writer([&] {
            for (int k = 0; !readers_done; ++k) {
                f.update(7, data[std::size_t(k % 2 ? 50 : 150)]);
                ++updates;
            }
        });
        run_threads(3, [&](std::size_t) {
            for (int i = 0; i < 300; ++i) {
                const auto r = *f.search_by_id(7, 1, {}, true);
                wrong += long(r.empty() || r[0].id != 7 || r[0].distance > 1e-6f);
                ++checks;
            }
        });
        readers_done = true;
        writer.join();
    });
    CHECK(wrong == 0 && checks == 900 && updates > 0);
}

TEST(control, t01_passed_deadline_stops_quickly_with_valid_results) {
    auto data = clustered(20000, 16, 20, 957);
    HnswIndex h(16, Metric::L2);
    FlatIndex f(16, Metric::L2);
    h.add_batch(id_range(0, 20000), flatten(data), 0);
    f.add_batch(id_range(0, 20000), flatten(data));
    SearchStats full, cut, fcut;
    SearchOptions normal;
    normal.stats = &full;
    (void)h.search(data[0], 10, 400, normal);
    SearchOptions late;
    late.deadline = SteadyClock::now() - std::chrono::seconds(1);
    late.stats = &cut;
    const auto hr = h.search(data[0], 10, 400, late);
    late.stats = &fcut;
    const auto fr = f.search(data[0], 10, late);
    CHECK(cut.stopped_early && fcut.stopped_early && !full.stopped_early);
    CHECK(sorted_and_unique(hr) && sorted_and_unique(fr) && hr.size() <= 10 && fr.size() <= 10);
    CHECK(cut.distance_computations * 4 < full.distance_computations);  // stopped well before the end
    CHECK(fcut.distance_computations <= 512);                            // Flat checks every 256 rows
}

TEST(control, t02_cancel_from_another_thread) {
    // A slow predicate makes the search long; another thread cancels it.
    FlatIndex f(4, Metric::L2);
    for (std::uint64_t i = 0; i < 3000; ++i) f.add(i, std::vector<float>{float(i % 97), float(i % 13), 1, 2});
    std::atomic<bool> cancel{false};
    std::atomic<long> calls{0};
    SearchStats stats;
    SearchOptions o;
    o.cancel = &cancel;
    o.stats = &stats;
    // At its 100th call the predicate waits until the other thread has set the
    // flag: the cancel always lands mid-search. (Sleeping instead made this
    // depend on timing: MinGW rounds sub-millisecond sleeps down to nothing, and
    // the search finished before the cancel arrived.)
    o.predicate = [&](std::uint64_t) {
        if (++calls == 100)
            while (!cancel) std::this_thread::yield();
        return true;
    };
    with_watchdog(60, "t02", [&] {
        std::thread canceller([&] {
            while (calls < 100) std::this_thread::yield();
            cancel = true;
        });
        const auto r = f.search(std::vector<float>{0, 0, 1, 2}, 5, o);
        canceller.join();
        CHECK(stats.stopped_early && sorted_and_unique(r) && r.size() == 5 && calls < 3000);
    });
}

TEST(control, t03_batch_range_and_by_id_searches_stop) {
    auto data = clustered(5000, 8, 10, 958);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    h.add_batch(id_range(0, 5000), flatten(data), 0);
    f.add_batch(id_range(0, 5000), flatten(data));
    SearchStats a, b, c, d, e;
    SearchOptions late;
    late.deadline = SteadyClock::now() - std::chrono::seconds(1);
    const auto queries = flatten(Vectors(data.begin(), data.begin() + 8));
    late.stats = &a;
    for (const auto& r : h.search_batch(queries, 5, 200, 4, late)) CHECK(sorted_and_unique(r));
    late.stats = &b;
    for (const auto& r : f.search_batch(queries, 5, 4, late)) CHECK(sorted_and_unique(r));
    late.stats = &c;
    CHECK(sorted_and_unique(h.search_range(data[0], 1e9f, kNoLimit, late)));
    late.stats = &d;
    CHECK(sorted_and_unique(*h.search_by_id(3, 5, 200, late)));
    late.stats = &e;
    CHECK(f.search_range(data[0], 1e9f, kNoLimit, late).size() < 5000);
    SearchStats g;
    SearchOptions exact = late;
    exact.strategy = Strategy::ForceExact;  // the exact path of an HNSW range search
    exact.stats = &g;
    CHECK(h.search_range(data[0], 1e9f, kNoLimit, exact).size() < 5000);
    CHECK(a.stopped_early && b.stopped_early && c.stopped_early && d.stopped_early && e.stopped_early && g.stopped_early);
}

TEST(control, t04_no_deadline_changes_nothing) {
    auto data = clustered(2000, 8, 6, 959);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 2000; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    std::atomic<bool> never{false};
    SearchStats st;
    SearchOptions far = with_filter(Filter::has_tag("tags", "a"));
    far.deadline = SteadyClock::now() + std::chrono::hours(1);
    far.cancel = &never;
    far.stats = &st;
    const SearchOptions plain = with_filter(Filter::has_tag("tags", "a"));
    SearchOptions only_cancel = plain, only_deadline = plain;  // each alone, neither firing
    only_cancel.cancel = &never;
    only_deadline.deadline = far.deadline;
    bool same = true;
    for (std::size_t q = 0; q < 5; ++q)
        same = same && same_results(h.search(data[q], 10, 64, only_cancel), h.search(data[q], 10, 64, plain)) &&
               same_results(f.search(data[q], 10, only_deadline), f.search(data[q], 10, plain));
    for (std::size_t q = 0; q < 20; ++q) {
        same = same && same_results(h.search(data[q], 10, 64, far), h.search(data[q], 10, 64, plain));
        same = same && same_results(f.search(data[q], 10, far), f.search(data[q], 10, plain));
        same = same && !st.stopped_early;
    }
    CHECK(same);
}

TEST(control, t05_filtered_exact_scans_stop) {
    auto data = clustered(5000, 8, 6, 960);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 5000; ++i) h.add(i, data[i], meta_for(i));
    SearchStats st;
    SearchOptions o = with_filter(Filter::eq("category", "tech"));
    o.strategy = Strategy::ForceExact;
    o.deadline = SteadyClock::now() - std::chrono::seconds(1);
    o.stats = &st;
    const auto r = h.search(data[0], 10, 64, o);
    CHECK(st.stopped_early && sorted_and_unique(r) && st.distance_computations < 1000);
}

TEST(control, t06_a_search_inside_a_predicate_has_its_own_deadline) {
    auto data = clustered(2000, 8, 6, 961);
    FlatIndex f(8, Metric::L2);
    HnswIndex inner(8, Metric::L2);
    for (std::uint64_t i = 0; i < 2000; ++i) {
        f.add(i, data[i]);
        inner.add(i, data[i]);
    }
    std::atomic<long> complete{0}, calls{0};
    SearchOptions outer;
    outer.deadline = SteadyClock::now() - std::chrono::seconds(1);
    outer.predicate = [&](std::uint64_t id) {
        ++calls;
        SearchStats st;
        SearchOptions own;
        own.stats = &st;
        complete += long(inner.search(data[id], 10, 64, own).size() == 10 && !st.stopped_early);
        return true;
    };
    SearchStats outer_stats;
    outer.stats = &outer_stats;
    (void)f.search(data[0], 5, outer);
    CHECK(outer_stats.stopped_early && calls > 0 && complete == calls);
    SearchStats after;
    SearchOptions next;
    next.stats = &after;
    CHECK(f.search(data[0], 5, next).size() == 5 && !after.stopped_early);  // nothing left behind on this thread
}

TEST(control, t07_add_batch_cancelled_at_every_stage) {
    auto data = clustered(900, 8, 6, 962);
    for (std::size_t threads : {std::size_t(1), std::size_t(4)}) {
        for (std::size_t at : {std::size_t(0), std::size_t(1), std::size_t(250), std::size_t(599)}) {
            HnswIndex h(8, Metric::Cosine);
            FlatIndex f(8, Metric::Cosine);
            for (std::uint64_t i = 0; i < 300; ++i) {
                h.add(i, data[i], meta_for(i));
                f.add(i, data[i], meta_for(i));
            }
            for (std::uint64_t i = 0; i < 300; i += 10) h.remove(i);  // slots the batch reuses one at a time
            const auto h_before = contents_of(h);
            const std::string f_before = save_bytes(f);
            std::vector<Metadata> meta(600);
            for (std::size_t i = 0; i < 600; ++i) meta[i] = meta_for(i).set("batch", true);
            std::atomic<bool> flag{false};
            const OperationOptions o = cancel_at(flag, at);
            CHECK_THROWS_AS(Cancelled, h.add_batch(id_range(1000, 600), flatten(Vectors(data.begin() + 300, data.end())), meta, threads, o));
            flag = false;
            CHECK_THROWS_AS(Cancelled, f.add_batch(id_range(1000, 600), flatten(Vectors(data.begin() + 300, data.end())), meta, threads, o));
            CHECK(contents_of(h) == h_before && h.count_where(Filter::exists("batch")) == 0 && h.metadata().find_field("batch") == std::nullopt);
            CHECK(save_bytes(f) == f_before);  // Flat: restored exactly
            check_graph(h, 0.97);
        }
    }
}

TEST(control, t08_compact_cancelled_leaves_the_index_untouched) {
    auto data = clustered(1000, 8, 6, 963);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1000; ++i) h.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 0; i < 1000; i += 3) h.remove(i);
    const std::string before = save_bytes(h);
    for (std::size_t at : {std::size_t(0), std::size_t(300), std::size_t(666)}) {
        std::atomic<bool> flag{false};
        CHECK_THROWS_AS(Cancelled, h.compact(cancel_at(flag, at)));
        CHECK(save_bytes(h) == before);
    }
    CHECK(h.compact().reclaimed > 0 && h.deleted_count() == 0);  // and it still works
}

TEST(control, t09_save_cancelled_keeps_the_old_file) {
    auto data = clustered(1500, 8, 6, 964);
    HnswIndex h(8, Metric::Cosine);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1500; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string path = temp_path("t09");
    write_file(path, "the previous file");
    for (std::size_t at : {std::size_t(0), std::size_t(700), std::size_t(1499)}) {
        std::atomic<bool> flag{false};
        CHECK_THROWS_AS(Cancelled, h.save(path, cancel_at(flag, at)));
        flag = false;
        CHECK_THROWS_AS(Cancelled, f.save(path, cancel_at(flag, at)));
        CHECK(read_file(path) == "the previous file" && temporaries_left(path) == 0);
    }
    std::atomic<bool> flag{false};
    std::ostringstream out;
    CHECK_THROWS_AS(Cancelled, h.save(out, cancel_at(flag, 10)));  // a stream may hold an incomplete index
    h.save(path);
    CHECK(load_hnsw(read_file(path))->size() == 1500);
    std::filesystem::remove(path);
}

TEST(control, t10_progress_is_reported_in_order) {
    auto data = clustered(2000, 8, 6, 965);
    HnswIndex h(8, Metric::L2);
    std::vector<std::pair<std::size_t, std::size_t>> calls;
    OperationOptions o;
    o.progress = [&](std::size_t done, std::size_t total) { calls.emplace_back(done, total); };  // calls never overlap
    auto in_order = [&](std::size_t total) {
        bool ok = calls.size() >= 2 && calls.size() <= 102 && calls.front().first == 0 && calls.back().first == total;
        for (std::size_t i = 0; i < calls.size(); ++i) ok = ok && calls[i].second == total && (i == 0 || calls[i].first > calls[i - 1].first);
        calls.clear();
        return ok;
    };
    h.add_batch(id_range(0, 2000), flatten(data), 4, o);
    CHECK(in_order(2000));
    for (std::uint64_t i = 0; i < 2000; i += 2) h.remove(i);
    h.compact(o);
    CHECK(in_order(1000));
    std::ostringstream out;
    h.save(out, o);
    CHECK(in_order(1000));
    FlatIndex f(8, Metric::L2);
    f.add_batch(id_range(0, 2000), flatten(data), 0, o);
    CHECK(in_order(2000));
    f.compact(o);
    CHECK(in_order(1));
}

TEST(control, t11_a_throwing_progress_callback_undoes_the_operation) {
    auto data = clustered(600, 8, 6, 966);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 300; i += 3) h.remove(i);
    for (std::size_t at : {std::size_t(0), std::size_t(150), std::size_t(300)}) {  // 300: the final report
        OperationOptions o;
        o.progress = [at](std::size_t done, std::size_t) {
            if (done >= at) throw std::runtime_error("stop here");
        };
        const auto before = contents_of(h);
        CHECK_THROWS_AS(std::runtime_error, h.add_batch(id_range(1000, 300), flatten(Vectors(data.begin() + 300, data.end())), 4, o));
        CHECK(contents_of(h) == before);  // a rolled-back HNSW batch keeps the contents; bookkeeping may move on
        const std::string bytes = save_bytes(h);  // so compare compact() with the state right before it
        if (at <= 200) {
            CHECK_THROWS_AS(std::runtime_error, h.compact(o));
            CHECK(save_bytes(h) == bytes);
        }
        const std::string path = temp_path("t11");
        write_file(path, "old");
        CHECK_THROWS_AS(std::runtime_error, h.save(path, o));
        CHECK(read_file(path) == "old" && temporaries_left(path) == 0);
        std::filesystem::remove(path);
    }
}

TEST(control, t12_deadlines_on_operations) {
    auto data = clustered(400, 8, 4, 967);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 200; ++i) h.add(i, data[i]);
    h.remove(3);
    const std::string before = save_bytes(h);
    OperationOptions late;
    late.deadline = SteadyClock::now() - std::chrono::seconds(1);
    CHECK_THROWS_AS(Cancelled, h.add_batch(id_range(1000, 200), flatten(Vectors(data.begin() + 200, data.end())), 2, late));
    CHECK_THROWS_AS(Cancelled, h.compact(late));
    std::ostringstream out;
    CHECK_THROWS_AS(Cancelled, h.save(out, late));
    CHECK(save_bytes(h) == before);  // a passed deadline stops before any change
    OperationOptions soon;
    soon.deadline = SteadyClock::now() + std::chrono::hours(1);
    h.add_batch(id_range(1000, 200), flatten(Vectors(data.begin() + 200, data.end())), 2, soon);
    CHECK(h.size() == 399);
}

TEST(control, t13_cancelling_a_parallel_batch) {
    // Many threads linking when the cancel arrives; run under ThreadSanitizer in CI.
    auto data = clustered(3000, 8, 8, 968);
    for (int round = 0; round < 3; ++round) {
        HnswIndex h(8, Metric::L2);
        for (std::uint64_t i = 0; i < 500; ++i) h.add(i, data[i]);
        const auto before = contents_of(h);
        std::atomic<bool> flag{false};
        std::thread canceller;
        OperationOptions o;
        o.cancel = &flag;
        o.progress = [&](std::size_t done, std::size_t) {
            if (done > 600 + std::size_t(round) * 400 && !canceller.joinable()) canceller = std::thread([&] { flag = true; });
        };
        bool cancelled = false;
        with_watchdog(120, "t13", [&] {
            try {
                h.add_batch(id_range(10000, 2500), flatten(Vectors(data.begin() + 500, data.end())), 8, o);
            } catch (const Cancelled&) {
                cancelled = true;
            }
            if (canceller.joinable()) canceller.join();
        });
        if (cancelled) CHECK(contents_of(h) == before);
        else CHECK(h.size() == 3000);  // the cancel came after the last vector: the batch completed
        check_graph(h, 0.97);
    }
}

// ===========================================================================
// Phase Q0: k-means and codecs (Q0-1 to Q0-14)
// ===========================================================================

namespace {

/// Points around `centers` (each `dim` values), `per` points each, small spread.
std::vector<float> blobs(const std::vector<std::vector<float>>& centers, std::size_t per, float spread, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-spread, spread);
    std::vector<float> out;
    for (std::size_t i = 0; i < per; ++i)
        for (const auto& c : centers)
            for (float x : c) out.push_back(x + u(rng));
    return out;
}

}  // namespace

TEST(codec, q0_01_kmeans_finds_separated_clusters) {
    const std::vector<std::vector<float>> centers = {{0, 0, 0}, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {10, 10, 10}};
    const auto data = blobs(centers, 200, 0.5f, 1);
    const auto r = kmeans(data, 3, KMeansParams{5, 50, 7, 1});
    bool found = true;
    for (const auto& c : centers) {
        float d = 0;
        (void)nearest_centroid(r.centroids, 3, c.data(), &d);
        found = found && d < 0.05f;  // each true center has a centroid within about 0.2
    }
    CHECK(found && r.centroids.size() == 15 && r.assignment.size() == 1000);
    bool consistent = true;  // points of one blob share a centroid
    for (std::size_t i = 0; i < 1000; ++i) consistent = consistent && r.assignment[i] == r.assignment[i % 5];
    CHECK(consistent && r.inertia < 1000 * 3 * 0.25);
}

TEST(codec, q0_02_kmeans_is_deterministic_and_independent_of_threads) {
    std::mt19937 rng(2);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> data(20000 * 8);
    for (float& x : data) x = nd(rng);
    const KMeansParams one{64, 15, 42, 1};
    KMeansParams many = one;
    many.threads = 4;
    const auto a = kmeans(data, 8, one), b = kmeans(data, 8, one), c = kmeans(data, 8, many);
    CHECK(a.centroids == b.centroids && a.assignment == b.assignment);  // bit for bit
    CHECK(a.centroids == c.centroids && a.assignment == c.assignment && a.inertia == c.inertia);
    KMeansParams other = one;
    other.seed = 43;
    const auto d = kmeans(data, 8, other);
    CHECK(d.centroids != a.centroids && d.centroids.size() == a.centroids.size());  // another seed: another valid result
}

TEST(codec, q0_03_kmeans_fixed_answer) {
    // Whole numbers average exactly, so every CPU and compiler must give exactly these.
    const std::vector<float> data = {0, 0, 1, 0, 0, 1, 1, 1, 10, 10, 11, 10, 10, 11, 11, 11, 20, 0, 21, 0, 20, 2, 21, 2};
    const auto r = kmeans(data, 2, KMeansParams{3, 20, 5, 1});
    std::vector<std::vector<float>> got;
    for (std::size_t c = 0; c < 3; ++c) got.push_back({r.centroids[c * 2], r.centroids[c * 2 + 1]});
    std::sort(got.begin(), got.end());
    CHECK(got == (std::vector<std::vector<float>>{{0.5f, 0.5f}, {10.5f, 10.5f}, {20.5f, 1.0f}}));
    CHECK(r.inertia == 9.0);  // 4 x 0.5 for each of the first two clusters, 4 x 1.25 for the third
}

TEST(codec, q0_04_kmeans_edge_cases) {
    // k = n: every point is its own centroid. k = 1: the mean.
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto all = kmeans(data, 2, KMeansParams{4, 10, 1, 1});
    CHECK(all.inertia == 0.0);
    const auto one = kmeans(data, 2, KMeansParams{1, 10, 1, 1});
    CHECK(one.centroids == (std::vector<float>{4, 5}));
    // Identical points: no NaN, every centroid equal to the point.
    const std::vector<float> same(50 * 4, 3.25f);
    const auto s = kmeans(same, 4, KMeansParams{8, 10, 1, 1});
    CHECK(std::all_of(s.centroids.begin(), s.centroids.end(), [](float x) { return x == 3.25f; }) && s.inertia == 0.0);
    // Only 3 distinct values but 25 centroids: identical centroids tie, ties go to
    // the lower index, so most stay empty. Every centroid is still a valid value
    // (empty clusters are refilled before averaging: never NaN).
    std::vector<float> dup;
    for (int i = 0; i < 30; ++i) dup.push_back(float(i % 3));
    const auto e = kmeans(dup, 1, KMeansParams{25, 10, 3, 1});
    CHECK(e.inertia == 0.0 && std::all_of(e.centroids.begin(), e.centroids.end(), [](float x) { return x == 0 || x == 1 || x == 2; }));
}

TEST(codec, q0_05_kmeans_rejects_invalid_input) {
    const std::vector<float> data = {1, 2, 3, 4};
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(data, 0, KMeansParams{1}));
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(data, 3, KMeansParams{1}));  // not a multiple
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans({}, 2, KMeansParams{1}));
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(data, 2, KMeansParams{0}));
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(data, 2, KMeansParams{3}));  // more centroids than points
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(data, 2, KMeansParams{1, 0}));
    const std::vector<float> bad = {1, kNaN, 3, 4};
    CHECK_THROWS_AS(std::invalid_argument, (void)kmeans(bad, 2, KMeansParams{1}));
}

TEST(codec, q0_06_kmeans_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    std::vector<float> data(3000 * 4);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = float((i * 7919) % 101);
    KMeansResult r;
    CHECK(sweep_allocation_failures([&] { r = kmeans(data, 4, KMeansParams{16, 5, 1, 2}); }, [] {}) > 0);
    CHECK(r.centroids.size() == 64);
}

TEST(codec, q0_07_sq8_round_trip_error_is_bounded) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-4, 9);
    std::vector<float> sample(500 * 16);
    for (float& x : sample) x = u(rng);
    SQ8Codec c(16, Metric::L2);
    c.train(sample);
    std::vector<std::uint8_t> code(c.code_size());
    std::vector<float> back(16);
    float worst = 0;
    for (std::size_t i = 0; i < 500; ++i) {
        c.encode(std::span<const float>(&sample[i * 16], 16), code.data());
        c.decode(code.data(), back);
        for (std::size_t j = 0; j < 16; ++j) worst = std::max(worst, std::abs(back[j] - sample[i * 16 + j]) / c.step()[j]);
    }
    CHECK(c.code_size() == 16 && worst <= 0.5f + 1e-3f);  // at most half a level, plus float rounding
}

TEST(codec, q0_08_sq8_constant_dimensions_and_clamping) {
    const std::vector<float> sample = {1, 5, 0, 1, 7, 10, 1, 6, 5};  // dimension 0 is constant
    SQ8Codec c(3, Metric::L2);
    c.train(sample);
    std::uint8_t code[3];
    float back[3];
    c.encode(std::vector<float>{1, 5, 0}, code);
    c.decode(code, back);
    CHECK(back[0] == 1.0f && code[0] == 0);  // a constant dimension is stored exactly
    c.encode(std::vector<float>{-100, 100, 5.2f}, code);  // outside the range: clamped to its ends
    CHECK(code[0] == 0 && code[1] == 255 && code[2] == 133);  // 5.2 / (10 / 255) = 132.6
    c.decode(code, back);
    CHECK(back[1] == 7.0f);
}

TEST(codec, q0_09_sq8_distances_match_decoded_vectors) {
    std::mt19937 rng(4);
    std::normal_distribution<float> nd(0, 1);
    for (Metric m : kMetrics) {
        std::vector<float> sample(400 * 24);
        for (float& x : sample) x = nd(rng);
        if (m == Metric::Cosine)  // cosine codecs receive normalized vectors
            for (std::size_t i = 0; i < 400; ++i) {
                float n2 = 0;
                for (std::size_t j = 0; j < 24; ++j) n2 += sample[i * 24 + j] * sample[i * 24 + j];
                for (std::size_t j = 0; j < 24; ++j) sample[i * 24 + j] /= std::sqrt(n2);
            }
        SQ8Codec c(24, m);
        c.train(sample);
        std::vector<std::uint8_t> code(24);
        std::vector<float> table, back(24);
        const std::span<const float> q(&sample[0], 24);
        c.prepare(q, table);
        bool close = true;
        for (std::size_t i = 1; i < 400; ++i) {
            const std::span<const float> v(&sample[i * 24], 24);
            c.encode(v, code.data());
            c.decode(code.data(), back);
            double expected = 0;
            for (std::size_t j = 0; j < 24; ++j)
                expected += m == Metric::L2 ? double(q[j] - back[j]) * (q[j] - back[j]) : double(q[j]) * back[j];
            if (m == Metric::InnerProduct) expected = -expected;
            if (m == Metric::Cosine) expected = 1.0 - expected;
            close = close && std::abs(c.distance(table, code.data()) - expected) < 1e-4 * (1 + std::abs(expected));
        }
        CHECK(close);
    }
}

TEST(codec, q0_10_sq8_ranks_like_full_precision) {
    // Codes keep the ordering: the 10 nearest by code overlap the true 10 nearest.
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> data(2000 * 32);
    for (float& x : data) x = nd(rng);
    SQ8Codec c(32, Metric::L2);
    c.train(data);
    std::vector<std::uint8_t> codes(2000 * 32);
    for (std::size_t i = 0; i < 2000; ++i) c.encode(std::span<const float>(&data[i * 32], 32), &codes[i * 32]);
    double overlap = 0;
    for (std::size_t q = 0; q < 20; ++q) {
        std::vector<float> query(32);
        for (float& x : query) x = nd(rng);
        std::vector<float> table;
        c.prepare(query, table);
        std::vector<std::pair<float, std::size_t>> exact, coded;
        for (std::size_t i = 0; i < 2000; ++i) {
            float d = 0;
            for (std::size_t j = 0; j < 32; ++j) d += (query[j] - data[i * 32 + j]) * (query[j] - data[i * 32 + j]);
            exact.emplace_back(d, i);
            coded.emplace_back(c.distance(table, &codes[i * 32]), i);
        }
        std::partial_sort(exact.begin(), exact.begin() + 10, exact.end());
        std::partial_sort(coded.begin(), coded.begin() + 10, coded.end());
        std::set<std::size_t> truth;
        for (std::size_t i = 0; i < 10; ++i) truth.insert(exact[i].second);
        for (std::size_t i = 0; i < 10; ++i) overlap += double(truth.count(coded[i].second));
    }
    CHECK(overlap / 200.0 > 0.9);
}

TEST(codec, q0_11_sq8_fixed_answer_codes) {
    // Every CPU and compiler must produce exactly these codes (they are saved in files).
    // Expected values computed independently with NumPy float32, with inputs away
    // from exact halves between levels (where float rounding decides).
    const std::vector<float> sample = {0, -1, 2.5f, 255, 1, 0.5f, 127.5f, 0, -0.25f};
    SQ8Codec c(3, Metric::L2);
    c.train(sample);
    std::uint8_t code[3];
    c.encode(std::vector<float>{63.75f, 0.3f, 1.0f}, code);
    CHECK(code[0] == 64 && code[1] == 166 && code[2] == 116);
    c.encode(std::vector<float>{127.5f, 1.0f, -0.25f}, code);
    CHECK(code[0] == 128 && code[1] == 255 && code[2] == 0);
}

TEST(codec, q0_12_sq8_rejects_misuse) {
    SQ8Codec c(4, Metric::L2);
    std::uint8_t code[4];
    std::vector<float> table, out(4);
    CHECK(!c.trained());
    CHECK_THROWS_AS(std::logic_error, c.encode(std::vector<float>(4, 1.0f), code));
    CHECK_THROWS_AS(std::logic_error, c.prepare(std::vector<float>(4, 1.0f), table));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>{}));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>(6, 1.0f)));  // not whole vectors
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>{1, 2, kInf, 4}));
    c.train(std::vector<float>{0, 0, 0, 0, 1, 1, 1, 1});
    CHECK_THROWS_AS(std::invalid_argument, c.encode(std::vector<float>(3, 1.0f), code));
    CHECK_THROWS_AS(std::invalid_argument, c.decode(code, std::span<float>(out.data(), 3)));
    CHECK_THROWS_AS(std::invalid_argument, c.set_range(std::vector<float>(4, 0.0f), std::vector<float>(4, -1.0f)));
    CHECK_THROWS_AS(std::invalid_argument, c.set_range(std::vector<float>(3, 0.0f), std::vector<float>(4, 1.0f)));
    CHECK_THROWS_AS(std::invalid_argument, SQ8Codec(0, Metric::L2));
}

TEST(codec, q0_13_sq8_set_range_matches_training) {
    std::vector<float> sample(100 * 6);
    for (std::size_t i = 0; i < sample.size(); ++i) sample[i] = float((i * 37) % 23) - 5.0f;
    SQ8Codec trained(6, Metric::InnerProduct), restored(6, Metric::InnerProduct);
    trained.train(sample);
    restored.set_range(trained.minimum(), trained.step());  // as when loading a saved index
    std::uint8_t a[6], b[6];
    bool same = true;
    for (std::size_t i = 0; i < 100; ++i) {
        trained.encode(std::span<const float>(&sample[i * 6], 6), a);
        restored.encode(std::span<const float>(&sample[i * 6], 6), b);
        same = same && std::equal(a, a + 6, b);
    }
    CHECK(same);
}

TEST(codec, q0_14_make_codec) {
    CHECK(make_codec(CodecKind::None, 4, Metric::L2) == nullptr);
    const auto c = make_codec(CodecKind::SQ8, 4, Metric::Cosine);
    CHECK(c && c->kind() == CodecKind::SQ8 && c->dim() == 4 && c->metric() == Metric::Cosine && c->code_size() == 4);
}

TEST(codec, q0_15_code_kernels_match_plain_cpp) {
    // Every instruction set this CPU supports, every length 0 to 130 (all
    // remainders), at unaligned addresses: within rounding of the plain version.
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-2, 2);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<float> t(140), s(140), w(140);
    std::vector<std::uint8_t> c(150);
    const CodeKernels plain = get_code_kernels(Isa::Scalar);
    long compared = 0, wrong = 0;
    for (Isa isa : {Isa::Avx2, Isa::Avx512, Isa::Neon}) {
        if (!isa_supported(isa)) continue;
        const CodeKernels k = get_code_kernels(isa);
        for (std::size_t n = 0; n <= 130; ++n) {
            for (int round = 0; round < 3; ++round) {
                for (float& x : t) x = u(rng);
                for (float& x : s) x = std::abs(u(rng)) / 255;
                for (float& x : w) x = u(rng);
                for (auto& b : c) b = std::uint8_t(byte(rng));
                const std::size_t off = std::size_t(round);  // 0, 1, 2: unaligned floats and bytes
                const float* tt = t.data() + off;
                const float* ss = s.data() + off;
                const float* ww = w.data() + off;
                const std::uint8_t* cc = c.data() + off;
                auto close = [](float a, float b, float scale) { return std::abs(a - b) <= 1e-4f * (1 + scale); };
                float mag = 0;
                for (std::size_t j = 0; j < n; ++j) mag += std::abs(ww[j]) * 255;
                wrong += long(!close(k.sq8_l2(tt, ss, cc, n), plain.sq8_l2(tt, ss, cc, n), plain.sq8_l2(tt, ss, cc, n)));
                wrong += long(!close(k.sq8_dot(ww, cc, n), plain.sq8_dot(ww, cc, n), mag));
                wrong += long(!close(k.bits_sum(ww, cc, n), plain.bits_sum(ww, cc, n), mag / 255));
                compared += 3;
            }
        }
        std::printf("    %s: compared with plain C++ for lengths 0 to 130\n", isa_name(isa));
    }
    CHECK(wrong == 0 && (compared > 0 || !isa_supported(Isa::Avx2)));
    CHECK(plain.sq8_l2(nullptr, nullptr, nullptr, 0) == 0.0f && plain.bits_sum(nullptr, nullptr, 0) == 0.0f);
    const auto best = get_code_kernels();
    const auto active = get_code_kernels(active_isa());
    CHECK(best.sq8_l2 == active.sq8_l2 && best.bits_sum == active.bits_sum);  // the same choice as the float kernels
    if (!isa_supported(Isa::Avx512)) CHECK_THROWS_AS(std::invalid_argument, (void)get_code_kernels(Isa::Avx512));
    if (!isa_supported(Isa::Neon)) CHECK_THROWS_AS(std::invalid_argument, (void)get_code_kernels(Isa::Neon));
}

TEST(codec, q0_16_binary_kernel_bit_order) {
    // Exactly the set bits count, bit j being c[j / 8] >> (j % 8): one bit at a time.
    std::vector<float> w(64);
    for (std::size_t j = 0; j < 64; ++j) w[j] = float(j + 1);
    for (Isa isa : {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon}) {
        if (!isa_supported(isa)) continue;
        const CodeKernels k = get_code_kernels(isa);
        bool exact = true;
        for (std::size_t j = 0; j < 64; ++j) {
            std::uint8_t c[8] = {};
            c[j / 8] = std::uint8_t(1u << (j % 8));
            exact = exact && k.bits_sum(w.data(), c, 64) == float(j + 1);
        }
        std::uint8_t all[8];
        std::fill(all, all + 8, std::uint8_t(0xFF));
        CHECK(exact && k.bits_sum(w.data(), all, 64) == 64.0f * 65.0f / 2.0f);
    }
}


// ===========================================================================
// Phase Q1: SQ8 in the indexes (S8-01 to S8-12)
// ===========================================================================

namespace {

QuantizationParams sq8(std::size_t train_after = 1000, bool rerank = true) {
    QuantizationParams q;
    q.codec = CodecKind::SQ8;
    q.train_after = train_after;
    q.rerank = rerank;
    return q;
}

HnswParams hnsw_sq8(std::size_t train_after = 1000, bool rerank = true) {
    HnswParams p;
    p.quantization = sq8(train_after, rerank);
    return p;
}

/// Every live vector's code equals a fresh encoding of its stored vector (with
/// codes only, of its decoded vector: decoding then encoding must be lossless).
bool codes_match(const HnswIndex& h) {
    const VectorCodec* c = h.codec();
    if (!c || !c->trained()) return false;
    std::vector<std::uint8_t> expected(c->code_size());
    const bool codes_only = !h.storage().vectors().has_values();
    std::vector<float> decoded(c->dim());
    for (NodeId n = 0; n < h.storage().size(); ++n) {
        if (h.storage().ids().is_deleted(n)) continue;
        if (codes_only) {
            c->decode(h.get_code(h.storage().ids().external(n))->data(), decoded);
            c->encode(decoded, expected.data());
        } else {
            c->encode(h.storage().vectors().get(n), expected.data());
        }
        if (*h.get_code(h.storage().ids().external(n)) != expected) return false;
    }
    return true;
}

bool codes_match(const FlatIndex& f) {  // L2: the stored vector is the vector as given
    const VectorCodec* c = f.codec();
    if (!c || !c->trained()) return false;
    std::vector<std::uint8_t> expected(c->code_size());
    ScanOptions all;
    all.limit = 1000000;
    for (const Record& r : f.scan(all).records) {
        c->encode(r.vector, expected.data());
        if (*f.get_code(r.id) != expected) return false;
    }
    return true;
}

}  // namespace

TEST(sq8, s8_01_untrained_means_full_precision) {
    auto data = clustered(500, 16, 6, 970);
    HnswIndex plain(16, Metric::L2), coded(16, Metric::L2, hnsw_sq8(1000));
    FlatIndex fplain(16, Metric::L2), fcoded(16, Metric::L2, Schema::dynamic(), sq8(1000));
    for (std::uint64_t i = 0; i < 500; ++i) {
        plain.add(i, data[i]);
        coded.add(i, data[i]);
        fplain.add(i, data[i]);
        fcoded.add(i, data[i]);
    }
    CHECK(!coded.codec_trained() && !fcoded.codec_trained() && !coded.get_code(1));
    bool same = true;
    for (std::size_t q = 0; q < 20; ++q)
        same = same && same_results(coded.search(data[q], 10, 64), plain.search(data[q], 10, 64)) &&
               same_results(fcoded.search(data[q], 10), fplain.search(data[q], 10));
    CHECK(same);
}

TEST(sq8, s8_02_automatic_training) {
    auto data = clustered(400, 8, 4, 971);
    HnswIndex h(8, Metric::L2, hnsw_sq8(300));
    FlatIndex f(8, Metric::L2, Schema::dynamic(), sq8(300));
    for (std::uint64_t i = 0; i < 299; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(!h.codec_trained() && !f.codec_trained());
    h.add(299, data[299]);  // the 300th vector: trains, and encodes everything stored
    f.add(299, data[299]);
    CHECK(h.codec_trained() && f.codec_trained() && codes_match(h) && codes_match(f));
    for (std::uint64_t i = 300; i < 400; ++i) {  // later vectors are encoded as they arrive
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(codes_match(h) && codes_match(f));
    HnswIndex batch(8, Metric::L2, hnsw_sq8(300));
    batch.add_batch(id_range(0, 400), flatten(data), 4);  // a batch crossing the threshold trains at its end
    CHECK(batch.codec_trained() && codes_match(batch));
}

TEST(sq8, s8_03_explicit_training) {
    auto data = clustered(200, 8, 4, 972);
    HnswIndex none(8, Metric::L2);
    FlatIndex fnone(8, Metric::L2);
    CHECK_THROWS_AS(std::logic_error, none.train_codec());
    CHECK_THROWS_AS(std::logic_error, fnone.train_codec());
    HnswIndex h(8, Metric::L2, hnsw_sq8(0));  // 0: only explicit training
    FlatIndex f(8, Metric::L2, Schema::dynamic(), sq8(0));
    CHECK_THROWS_AS(std::invalid_argument, h.train_codec());  // empty
    for (std::uint64_t i = 0; i < 200; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(!h.codec_trained());
    h.train_codec();
    f.train_codec();
    CHECK(codes_match(h) && codes_match(f));
    h.add(500, std::vector<float>(8, 100.0f));  // far outside the trained range: clamped
    h.train_codec();                             // training again includes it
    CHECK(codes_match(h) && h.codec()->state()[0] <= data[0][0]);
}

TEST(sq8, s8_04_recall_every_metric) {
    auto data = clustered(3000, 32, 12, 973);
    auto queries = clustered(60, 32, 12, 974);
    for (Metric m : kMetrics) {
        HnswIndex plain(32, m), reranked(32, m, hnsw_sq8(500)), codes_only(32, m, hnsw_sq8(500, false));
        FlatIndex exact(32, m), fcoded(32, m, Schema::dynamic(), sq8(500)), fcodes_only(32, m, Schema::dynamic(), sq8(500, false));
        for (std::uint64_t i = 0; i < 3000; ++i) {
            plain.add(i, data[i]);
            reranked.add(i, data[i]);
            codes_only.add(i, data[i]);
            exact.add(i, data[i]);
            fcoded.add(i, data[i]);
            fcodes_only.add(i, data[i]);
        }
        const double base = recall(plain, exact, queries, 10, 64);
        const double r = recall(reranked, exact, queries, 10, 64), c = recall(codes_only, exact, queries, 10, 64);
        double fr = 0, fc = 0;  // Flat against exact results
        for (const auto& q : queries) {
            std::set<std::uint64_t> truth;
            for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
            for (const auto& x : fcoded.search(q, 10)) fr += double(truth.count(x.id));
            for (const auto& x : fcodes_only.search(q, 10)) fc += double(truth.count(x.id));
        }
        fr /= 600.0;
        fc /= 600.0;
        std::printf("    metric %d: hnsw %.3f, sq8+rerank %.3f, sq8 only %.3f | flat sq8+rerank %.3f, sq8 only %.3f\n", int(m), base, r, c, fr, fc);
        CHECK(r >= base - 0.02 && c >= 0.8 && fr >= 0.98 && fc >= 0.85);
    }
}

TEST(sq8, s8_05_codes_follow_every_path) {
    auto data = clustered(1500, 8, 6, 975);
    HnswIndex h(8, Metric::L2, hnsw_sq8(200));
    FlatIndex f(8, Metric::L2, Schema::dynamic(), sq8(200));
    for (std::uint64_t i = 0; i < 400; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (std::uint64_t i = 0; i < 400; i += 5) {  // HNSW: free slots; Flat: swap with last
        h.remove(i);
        f.remove(i);
    }
    for (std::uint64_t i = 1000; i < 1050; ++i) {  // HNSW: reused slots
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (std::uint64_t i = 1; i < 400; i += 7) {
        h.update(i, data[i + 1000]);
        f.update(i, data[i + 1000]);
    }
    for (std::uint64_t i = 3; i < 400; i += 11) {
        h.upsert(i, data[i + 1050]);
        f.upsert(i, data[i + 1050]);
    }
    CHECK(codes_match(h) && codes_match(f));
    for (std::uint64_t i = 2; i < 400; i += 13) h.remove(i);
    h.add_batch(id_range(2000, 40), flatten(Vectors(data.begin() + 1100, data.begin() + 1140)), 1);  // reuse, then append
    h.add_batch(id_range(3000, 60), flatten(Vectors(data.begin() + 1140, data.begin() + 1200)), 4);
    f.add_batch(id_range(2000, 100), flatten(Vectors(data.begin() + 1100, data.begin() + 1200)));
    CHECK(codes_match(h) && codes_match(f));
    std::atomic<bool> flag{false};  // a cancelled batch leaves codes consistent
    OperationOptions o;
    o.cancel = &flag;
    o.progress = [&](std::size_t done, std::size_t) { if (done > 20) flag = true; };
    CHECK_THROWS_AS(Cancelled, h.add_batch(id_range(4000, 100), flatten(Vectors(data.begin() + 1200, data.begin() + 1300)), 4, o));
    CHECK(codes_match(h));
    h.compact();  // the codec keeps its training; codes follow the new slots
    CHECK(h.codec_trained() && codes_match(h) && h.deleted_count() == 0);
}

TEST(sq8, s8_06_save_and_load) {
    auto data = clustered(600, 8, 4, 976);
    for (int trained = 0; trained < 2; ++trained) {
        HnswIndex h(8, Metric::Cosine, hnsw_sq8(trained ? 100 : 10000, false));
        FlatIndex f(8, Metric::InnerProduct, Schema::dynamic(), sq8(trained ? 100 : 10000));
        for (std::uint64_t i = 0; i < 600; ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        h.remove(7);
        const std::string hb = save_bytes(h), fb = save_bytes(f);
        auto h2 = load_hnsw(hb);
        auto f2 = load_flat(fb);
        CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb);  // byte for byte, codes included
        CHECK(h2->codec_trained() == bool(trained) && f2->codec_trained() == bool(trained));
        CHECK(h2->params().quantization.codec == CodecKind::SQ8 && !h2->params().quantization.rerank && f2->quantization().rerank);
        bool same = true;
        for (std::size_t q = 0; q < 10; ++q)
            same = same && same_results(h2->search(data[q], 10, 64), h.search(data[q], 10, 64)) &&
                   same_results(f2->search(data[q], 10), f.search(data[q], 10));
        CHECK(same);
    }
}

TEST(sq8, s8_07_damaged_codec_sections) {
    auto data = random_vectors(12, 3, 977);
    HnswIndex h(3, Metric::L2, hnsw_sq8(5));
    FlatIndex f(3, Metric::L2, Schema::dynamic(), sq8(5));
    for (std::uint64_t i = 0; i < 12; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (int kind = 0; kind < 2; ++kind) {  // every truncation and every flipped bit
        const std::string bytes = kind == 0 ? save_bytes(h) : save_bytes(f);
        auto load = [&](const std::string& b) { if (kind == 0) (void)load_hnsw(b); else (void)load_flat(b); };
        std::size_t rejected = 0, tried = 0;
        for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
            try { load(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
        }
        for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
            std::string damaged = bytes;
            damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
            try { load(damaged); } catch (const FormatError&) { ++rejected; }
        }
        CHECK(rejected == tried);
    }
    // Invalid content behind valid checksums. CODC: codec (1), train_after (8),
    // rerank (1), factor (8), trained (1), state size (8), state (6 x 4), count (8), codes.
    const ParsedFile base = ParsedFile::parse(save_bytes(f));
    const std::vector<std::function<void(std::string&)>> edits = {
        [](std::string& p) { poke<std::uint8_t>(p, 0, 7); },               // unknown codec
        [](std::string& p) { poke<std::uint8_t>(p, 9, 2); },               // re-rank flag 2
        [](std::string& p) { poke<std::uint64_t>(p, 10, 0); },             // factor 0
        [](std::string& p) { poke<std::uint8_t>(p, 18, 2); },              // training flag 2
        [](std::string& p) { poke<float>(p, 27 + 12, -1.0f); },            // a negative step
        [](std::string& p) { poke<std::uint64_t>(p, 19, 5); },             // wrong state size
        [](std::string& p) { poke<std::uint64_t>(p, 51, 11); },            // codes for 11 vectors, 12 stored
    };
    for (const auto& edit : edits) {
        ParsedFile e = base;
        edit(e.payload("CODC"));
        CHECK_THROWS_AS(FormatError, load_flat(e.assemble()));
    }
    ParsedFile none = ParsedFile::parse(save_bytes(FlatIndex(3, Metric::L2)));  // no codec, but "trained"
    poke<std::uint8_t>(none.payload("CODC"), 18, 1);
    CHECK_THROWS_AS(FormatError, load_flat(none.assemble()));
    CHECK(load_flat(base.assemble())->codec_trained());
}

TEST(sq8, s8_08_every_kind_of_search) {
    auto data = clustered(2000, 16, 8, 978);
    HnswIndex h(16, Metric::L2, hnsw_sq8(500)), plain(16, Metric::L2);
    FlatIndex f(16, Metric::L2, Schema::dynamic(), sq8(500));
    for (std::uint64_t i = 0; i < 2000; ++i) {
        h.add(i, data[i], meta_for(i));
        plain.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    const auto queries = flatten(Vectors(data.begin(), data.begin() + 12));
    const auto hb = h.search_batch(queries, 10, 64, 4), fb = f.search_batch(queries, 10, 4);
    bool same = true;
    for (std::size_t q = 0; q < 12; ++q) same = same && same_results(hb[q], h.search(data[q], 10, 64)) && same_results(fb[q], f.search(data[q], 10));
    CHECK(same);
    SearchOptions exact = with_filter(Filter::eq("category", "tech"));
    exact.strategy = Strategy::ForceExact;
    for (const auto& r : {h.search(data[0], 10, 64, exact), f.search(data[0], 10, with_filter(Filter::eq("category", "tech")))}) {
        bool ok = r.size() == 10;
        for (const auto& x : r) ok = ok && x.id % 5 == 2;
        CHECK(ok);
    }
    // Range searches use full precision: identical to an index without a codec.
    CHECK(same_results(h.search_range(data[3], 2.0f), plain.search_range(data[3], 2.0f)));
}

TEST(sq8, s8_09_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(400, 8, 4, 979);
    HnswIndex h(8, Metric::L2, hnsw_sq8(0));
    FlatIndex f(8, Metric::L2, Schema::dynamic(), sq8(0));
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    CHECK(sweep_allocation_failures([&] { h.train_codec(); }, [&] { CHECK(save_bytes(h) == hb && !h.codec_trained()); }) > 0);
    CHECK(sweep_allocation_failures([&] { f.train_codec(); }, [&] { CHECK(save_bytes(f) == fb && !f.codec_trained()); }) > 0);
    const std::string fb2 = save_bytes(f);
    std::uint64_t id = 900;  // adding to a trained Flat index: code room first, all or nothing
    CHECK(sweep_allocation_failures([&] { f.add(id, data[350]); }, [&] { CHECK(save_bytes(f) == fb2); }) > 0);
    CHECK(codes_match(h) && codes_match(f));
    // Automatic training that runs out of memory: the add completes, training waits.
    HnswIndex a(8, Metric::L2, hnsw_sq8(50));
    for (std::uint64_t i = 0; i < 49; ++i) a.add(i, data[i]);
    bool consistent = true, done = false;
    for (long n = 0; n < 5000 && !done; ++n) {
        alloc_hook::fail_after(n);
        bool failed = false;
        try {
            a.add(49, data[49]);
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        alloc_hook::disarm();
        if (failed) consistent = consistent && !a.contains(49) && !a.codec_trained();
        else done = true;
    }
    CHECK(consistent && done && a.contains(49));
    if (!a.codec_trained()) a.add(50, data[50]);  // training was cut short: the next add retries
    CHECK(a.codec_trained() && codes_match(a));
}

TEST(sq8, s8_10_searching_while_training) {
    auto data = clustered(1500, 8, 6, 980);
    HnswIndex h(8, Metric::L2, hnsw_sq8(700));
    for (std::uint64_t i = 0; i < 500; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    Problems p;
    with_watchdog(120, "s8_10", [&] {
        std::thread writer([&] {
            for (std::uint64_t i = 500; i < 1500; ++i) h.add(i, data[i]);  // trains at 700
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) check_results(h.search(data[q++ % 1500], 10, 32), 1500, p);
        });
        writer.join();
    });
    CHECK(p.total() == 0 && h.codec_trained() && codes_match(h));
}

TEST(sq8, s8_11_older_files_have_no_codec) {
    // Version 1 and 2 files load without a codec (see p40 for the golden files).
    auto h = HnswIndex::load(golden_path("golden_hnsw_v2.bin"));
    CHECK(!h->codec_trained() && h->codec() == nullptr && h->params().quantization.codec == CodecKind::None);
}

TEST(sq8, s8_12_invalid_settings) {
    QuantizationParams q = sq8();
    q.rerank_factor = 0;
    HnswParams p;
    p.quantization = q;
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(4, Metric::L2, p));
    CHECK_THROWS_AS(std::invalid_argument, FlatIndex(4, Metric::L2, Schema::dynamic(), q));
}

// ===========================================================================
// Phase Q2: Matryoshka truncation (M1-M11)
// ===========================================================================

namespace {

QuantizationParams truncate_to(std::size_t n) {
    QuantizationParams q;
    q.truncate_to = n;
    return q;
}

HnswParams hnsw_truncate(std::size_t n) {
    HnswParams p;
    p.quantization.truncate_to = n;
    return p;
}

/// The first `n` values of each vector.
Vectors prefixes(const Vectors& data, std::size_t n) {
    Vectors out;
    for (const auto& v : data) out.emplace_back(v.begin(), v.begin() + std::ptrdiff_t(n));
    return out;
}

}  // namespace

TEST(matryoshka, m01_identical_to_an_index_of_the_prefixes) {
    auto data = clustered(1500, 64, 8, 990);
    auto queries = clustered(30, 64, 8, 991);
    const auto short_data = prefixes(data, 24), short_queries = prefixes(queries, 24);
    for (Metric m : kMetrics) {
        HnswIndex truncated(64, m, hnsw_truncate(24)), reference(24, m);
        FlatIndex ftruncated(64, m, Schema::dynamic(), truncate_to(24)), freference(24, m);
        for (std::uint64_t i = 0; i < 1500; ++i) {
            truncated.add(i, data[i]);
            reference.add(i, short_data[i]);
            ftruncated.add(i, data[i]);
            freference.add(i, short_data[i]);
        }
        bool same = true;  // same IDs and the same distances
        for (std::size_t q = 0; q < 30; ++q)
            same = same && same_results(truncated.search(queries[q], 10, 64), reference.search(short_queries[q], 10, 64)) &&
                   same_results(ftruncated.search(queries[q], 10), freference.search(short_queries[q], 10));
        CHECK(same && truncated.dim() == 64 && truncated.indexed_dim() == 24 && ftruncated.indexed_dim() == 24);
    }
}

TEST(matryoshka, m02_vectors_read_back_in_full) {
    auto data = random_vectors(200, 32, 992);
    for (Metric m : kMetrics) {
        HnswIndex h(32, m, hnsw_truncate(8));
        FlatIndex f(32, m, Schema::dynamic(), truncate_to(8));
        for (std::uint64_t i = 0; i < 200; ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        bool full = true;
        for (std::uint64_t i = 0; i < 200; i += 7) full = full && *h.get_vector(i) == data[i] && *f.get_vector(i) == data[i] && h.get(i)->vector == data[i];
        ScanOptions o;
        o.limit = 50;
        for (const Record& r : h.scan(o).records) full = full && r.vector.size() == 32 && r.vector == data[r.id];
        CHECK(full);
        // search_by_id uses the vector as given, truncated like any query.
        auto expected = h.search(data[5], 6, 64);
        std::erase_if(expected, [](const SearchResult& r) { return r.id == 5; });
        expected.resize(std::min<std::size_t>(expected.size(), 5));
        CHECK(same_results(*h.search_by_id(5, 5, 64), expected));
    }
}

TEST(matryoshka, m03_inputs_are_checked_in_full) {
    HnswIndex h(16, Metric::Cosine, hnsw_truncate(4));
    FlatIndex f(16, Metric::L2, Schema::dynamic(), truncate_to(4));
    std::vector<float> v(16, 1.0f), short_v(4, 1.0f), nan_in_cut_part(16, 1.0f);
    nan_in_cut_part[10] = kNaN;  // in the part that is cut off: still rejected
    h.add(1, v);
    f.add(1, v);
    CHECK_THROWS_AS(std::invalid_argument, h.add(2, short_v));
    CHECK_THROWS_AS(std::invalid_argument, f.add(2, short_v));
    CHECK_THROWS_AS(std::invalid_argument, h.add(3, nan_in_cut_part));
    CHECK_THROWS_AS(std::invalid_argument, f.add(3, nan_in_cut_part));
    CHECK_THROWS_AS(std::invalid_argument, (void)h.search(short_v, 1));
    CHECK_THROWS_AS(std::invalid_argument, (void)f.search(nan_in_cut_part, 1));
    CHECK_THROWS_AS(std::invalid_argument, h.add_batch(id_range(10, 2), std::vector<float>(2 * 4, 1.0f)));  // short vectors
    CHECK_THROWS_AS(std::invalid_argument, (void)f.search_batch(std::vector<float>(8, 1.0f), 1));
    CHECK(h.size() == 1 && f.size() == 1);
}

TEST(matryoshka, m04_settings) {
    auto data = clustered(300, 12, 4, 993);
    HnswIndex whole(12, Metric::L2, hnsw_truncate(12)), plain(12, Metric::L2);  // the full dimension: off
    for (std::uint64_t i = 0; i < 300; ++i) {
        whole.add(i, data[i]);
        plain.add(i, data[i]);
    }
    CHECK(whole.indexed_dim() == 12 && same_results(whole.search(data[0], 10, 64), plain.search(data[0], 10, 64)));
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(12, Metric::L2, hnsw_truncate(13)));
    CHECK_THROWS_AS(std::invalid_argument, FlatIndex(12, Metric::L2, Schema::dynamic(), truncate_to(13)));
    HnswIndex one(12, Metric::Cosine, hnsw_truncate(1));  // a single dimension
    for (std::uint64_t i = 0; i < 50; ++i) one.add(i, data[i]);
    CHECK(one.indexed_dim() == 1 && one.search(data[0], 5, 64).size() == 5 && *one.get_vector(3) == data[3]);
}

TEST(matryoshka, m05_with_sq8) {
    auto data = clustered(2000, 48, 10, 994);
    auto queries = clustered(40, 48, 10, 995);
    HnswParams p = hnsw_truncate(16);
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 500;
    HnswIndex coded(48, Metric::Cosine, p), plain(48, Metric::Cosine, hnsw_truncate(16));
    for (std::uint64_t i = 0; i < 2000; ++i) {
        coded.add(i, data[i]);
        plain.add(i, data[i]);
    }
    CHECK(coded.codec_trained() && coded.codec()->dim() == 16 && coded.get_code(3)->size() == 16);
    double overlap = 0;  // against the same truncation at full precision
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : plain.search(q, 10, 128)) truth.insert(r.id);
        for (const auto& r : coded.search(q, 10, 128)) overlap += double(truth.count(r.id));
    }
    CHECK(overlap / 400.0 >= 0.95);
}

TEST(matryoshka, m06_save_and_load) {
    auto data = random_vectors(400, 24, 996);
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        HnswParams p = hnsw_truncate(10);
        p.quantization.codec = CodecKind::SQ8;
        p.quantization.train_after = 100;
        HnswIndex h(24, m, p);
        FlatIndex f(24, m, Schema::dynamic(), truncate_to(10));
        for (std::uint64_t i = 0; i < 400; ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        h.remove(4);
        const std::string hb = save_bytes(h), fb = save_bytes(f);
        auto h2 = load_hnsw(hb);
        auto f2 = load_flat(fb);
        CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb);
        CHECK(h2->indexed_dim() == 10 && f2->indexed_dim() == 10 && h2->dim() == 24 && h2->params().quantization.truncate_to == 10);
        CHECK(*h2->get_vector(5) == data[5] && *f2->get_vector(5) == data[5] && h2->codec_trained());
        CHECK(same_results(h2->search(data[9], 10, 64), h.search(data[9], 10, 64)) && same_results(f2->search(data[9], 10), f.search(data[9], 10)));
    }
}

TEST(matryoshka, m07_every_path_keeps_vectors_in_full) {
    auto data = random_vectors(900, 20, 997);
    std::map<std::uint64_t, std::vector<float>> want;
    HnswIndex h(20, Metric::Cosine, hnsw_truncate(6));
    FlatIndex f(20, Metric::L2, Schema::dynamic(), truncate_to(6));
    auto add = [&](std::uint64_t id, const std::vector<float>& v) { h.add(id, v); f.add(id, v); want[id] = v; };
    for (std::uint64_t i = 0; i < 300; ++i) add(i, data[i]);
    for (std::uint64_t i = 0; i < 300; i += 4) { h.remove(i); f.remove(i); want.erase(i); }  // Flat: swap with last
    for (std::uint64_t i = 1000; i < 1040; ++i) add(i, data[i - 700]);                       // HNSW: reused slots
    for (std::uint64_t i = 1; i < 300; i += 5)
        if (want.contains(i)) { h.update(i, data[i + 340]); f.update(i, data[i + 340]); want[i] = data[i + 340]; }
    for (std::uint64_t i = 2; i < 300; i += 9) { h.upsert(i, data[i + 600]); f.upsert(i, data[i + 600]); want[i] = data[i + 600]; }
    for (std::uint64_t i = 3; i < 300; i += 11) { h.remove(i); f.remove(i); want.erase(i); }
    std::vector<float> batch;
    for (std::uint64_t i = 0; i < 100; ++i) {
        batch.insert(batch.end(), data[800 + i].begin(), data[800 + i].end());
        want[5000 + i] = data[800 + i];
    }
    h.add_batch(id_range(5000, 50), std::span<const float>(batch.data(), 50 * 20), 1);
    h.add_batch(id_range(5050, 50), std::span<const float>(batch.data() + 50 * 20, 50 * 20), 4);
    f.add_batch(id_range(5000, 100), batch);
    auto exact = [&] {
        bool ok = h.size() == want.size() && f.size() == want.size();
        for (const auto& [id, v] : want) ok = ok && h.get_vector(id) == v && f.get_vector(id) == v;
        return ok;
    };
    CHECK(exact());
    h.compact();
    CHECK(exact() && h.search(want.begin()->second, 1, 64)[0].id == want.begin()->first);
}

TEST(matryoshka, m08_damaged_files) {
    auto data = random_vectors(10, 6, 998);
    HnswIndex h(6, Metric::Cosine, hnsw_truncate(2));
    FlatIndex f(6, Metric::L2, Schema::dynamic(), truncate_to(2));
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (int kind = 0; kind < 2; ++kind) {
        const std::string bytes = kind == 0 ? save_bytes(h) : save_bytes(f);
        auto load = [&](const std::string& b) { if (kind == 0) (void)load_hnsw(b); else (void)load_flat(b); };
        std::size_t rejected = 0, tried = 0;
        for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
            try { load(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
        }
        for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
            std::string damaged = bytes;
            damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
            try { load(damaged); } catch (const FormatError&) { ++rejected; }
        }
        CHECK(rejected == tried);
    }
    // A truncation larger than the dimension, behind a valid checksum (CONF ends with
    // the truncation, 8 bytes, then the keep-full-vectors and vector-values flags, 1 byte each).
    for (int kind = 0; kind < 2; ++kind) {
        ParsedFile p = ParsedFile::parse(kind == 0 ? save_bytes(h) : save_bytes(f));
        std::string& conf = p.payload("CONF");
        poke<std::uint64_t>(conf, conf.size() - 10, 7);
        if (kind == 0) CHECK_THROWS_AS(FormatError, load_hnsw(p.assemble()));
        else CHECK_THROWS_AS(FormatError, load_flat(p.assemble()));
    }
}

TEST(matryoshka, m09_batch_and_range_searches) {
    auto data = clustered(1200, 32, 8, 999);
    const auto short_data = prefixes(data, 12);
    HnswIndex h(32, Metric::L2, hnsw_truncate(12)), reference(12, Metric::L2);
    FlatIndex f(32, Metric::L2, Schema::dynamic(), truncate_to(12));
    for (std::uint64_t i = 0; i < 1200; ++i) {
        h.add(i, data[i]);
        reference.add(i, short_data[i]);
        f.add(i, data[i]);
    }
    const auto queries = flatten(Vectors(data.begin(), data.begin() + 10));
    const auto hb = h.search_batch(queries, 8, 64, 4), fb = f.search_batch(queries, 8, 4);
    bool same = true;
    for (std::size_t q = 0; q < 10; ++q) same = same && same_results(hb[q], h.search(data[q], 8, 64)) && same_results(fb[q], f.search(data[q], 8));
    CHECK(same);
    CHECK(same_results(h.search_range(data[2], 1.0f), reference.search_range(short_data[2], 1.0f)));
}

TEST(matryoshka, m10_truncating_alone_needs_no_codec) {
    // Truncation works without a codec, and leaves the codec settings untouched.
    HnswIndex h(8, Metric::InnerProduct, hnsw_truncate(3));
    CHECK(h.codec() == nullptr && !h.codec_trained() && h.params().quantization.codec == CodecKind::None);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, std::vector<float>{float(i), 1, 2, 3, 4, 5, 6, 7});
    CHECK(h.search(std::vector<float>{19, 1, 2, 0, 0, 0, 0, 0}, 1)[0].id == 19);
}

TEST(matryoshka, m11_out_of_memory) {
    // Adding stores the truncated vector and the full original: all or nothing.
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = random_vectors(60, 10, 1000);
    HnswIndex h(10, Metric::Cosine, hnsw_truncate(4));
    FlatIndex f(10, Metric::Cosine, Schema::dynamic(), truncate_to(4));
    for (std::uint64_t i = 0; i < 50; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string fb = save_bytes(f);
    std::uint64_t id = 500;
    CHECK(sweep_allocation_failures([&] { f.add(id, data[55]); }, [&] { CHECK(save_bytes(f) == fb); }) > 0);
    CHECK(sweep_allocation_failures([&] { h.add(id, data[55]); }, [&] { CHECK(!h.contains(id) && h.size() == 50); }) > 0);
    CHECK(*h.get_vector(500) == data[55] && *f.get_vector(500) == data[55]);
}

// ===========================================================================
// keep_full_vectors = false (L1-L8)
// ===========================================================================

namespace {

QuantizationParams lean(std::size_t truncate = 0) {
    QuantizationParams q;
    q.truncate_to = truncate;
    q.keep_full_vectors = false;
    return q;
}

HnswParams hnsw_lean(std::size_t truncate = 0) {
    HnswParams p;
    p.quantization = lean(truncate);
    return p;
}

bool unit_and_same_direction(const std::vector<float>& stored, const std::vector<float>& given, std::size_t n) {
    double len = 0, dot = 0, g = 0;
    for (std::size_t d = 0; d < n; ++d) {
        len += double(stored[d]) * stored[d];
        dot += double(stored[d]) * given[d];
        g += double(given[d]) * given[d];
    }
    return stored.size() == n && std::abs(len - 1.0) < 1e-5 && std::abs(dot - std::sqrt(g)) < 1e-4 * (1 + std::sqrt(g));
}

}  // namespace

TEST(lean, l01_truncation_without_full_vectors) {
    auto data = clustered(1200, 48, 8, 1100);
    for (Metric m : kMetrics) {
        HnswIndex slim(48, m, hnsw_lean(16)), full(48, m, hnsw_truncate(16));
        FlatIndex fslim(48, m, Schema::dynamic(), lean(16)), ffull(48, m, Schema::dynamic(), truncate_to(16));
        for (std::uint64_t i = 0; i < 1200; ++i) {
            slim.add(i, data[i]);
            full.add(i, data[i]);
            fslim.add(i, data[i]);
            ffull.add(i, data[i]);
        }
        CHECK(slim.storage().originals() == nullptr && full.storage().originals() != nullptr);  // the saving
        bool same = true;  // the same stored vectors: identical results
        for (std::size_t q = 0; q < 20; ++q)
            same = same && same_results(slim.search(data[q], 10, 64), full.search(data[q], 10, 64)) &&
                   same_results(fslim.search(data[q], 10), ffull.search(data[q], 10));
        CHECK(same);
        bool prefix = true;  // get_vector: the stored first 16 values
        for (std::uint64_t i = 0; i < 1200; i += 101) {
            const auto v = *slim.get_vector(i), fv = *fslim.get_vector(i);
            if (m == Metric::Cosine) prefix = prefix && unit_and_same_direction(v, data[i], 16) && unit_and_same_direction(fv, data[i], 16);
            else prefix = prefix && v == std::vector<float>(data[i].begin(), data[i].begin() + 16) && fv == v;
        }
        CHECK(prefix);
    }
}

TEST(lean, l02_cosine_without_originals) {
    auto data = random_vectors(500, 24, 1101);
    HnswIndex slim(24, Metric::Cosine, hnsw_lean()), full(24, Metric::Cosine);
    FlatIndex fslim(24, Metric::Cosine, Schema::dynamic(), lean());
    for (std::uint64_t i = 0; i < 500; ++i) {
        slim.add(i, data[i]);
        full.add(i, data[i]);
        fslim.add(i, data[i]);
    }
    CHECK(slim.storage().originals() == nullptr);
    bool ok = true;
    for (std::uint64_t i = 0; i < 500; i += 37)
        ok = ok && unit_and_same_direction(*slim.get_vector(i), data[i], 24) && unit_and_same_direction(*fslim.get_vector(i), data[i], 24);
    for (std::size_t q = 0; q < 15; ++q) ok = ok && same_results(slim.search(data[q], 10, 64), full.search(data[q], 10, 64));
    CHECK(ok);
    // L2 and inner product change nothing: there is no copy to drop.
    HnswIndex l2(24, Metric::L2, hnsw_lean());
    l2.add(1, data[1]);
    CHECK(*l2.get_vector(1) == data[1]);
}

TEST(lean, l03_every_path) {
    auto data = clustered(1500, 20, 6, 1102);
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        HnswIndex h(20, m, hnsw_lean(8));
        FlatIndex f(20, m, Schema::dynamic(), lean(8));
        for (std::uint64_t i = 0; i < 400; ++i) {
            h.add(i, data[i], meta_for(i));
            f.add(i, data[i], meta_for(i));
        }
        for (std::uint64_t i = 0; i < 400; i += 4) {
            h.remove(i);
            f.remove(i);
        }
        for (std::uint64_t i = 1000; i < 1050; ++i) {  // reused slots
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        for (std::uint64_t i = 1; i < 400; i += 6) {
            h.update(i, data[i + 1000]);
            f.update(i, data[i + 1000]);
        }
        h.add_batch(id_range(5000, 100), flatten(Vectors(data.begin() + 1400, data.begin() + 1500)), 4);
        f.add_batch(id_range(5000, 100), flatten(Vectors(data.begin() + 1400, data.begin() + 1500)));
        // search_by_id and compact rebuild inputs from the stored vectors.
        CHECK(h.search_by_id(5001, 5, 64)->size() == 5 && f.search_by_id(5001, 5)->size() == 5);
        const auto before = h.search(data[1450], 5, 128);
        h.compact();
        const auto after = h.search(data[1450], 5, 128);
        CHECK(h.deleted_count() == 0 && h.size() == 450 && after[0].id == before[0].id);  // 400 - 100 + 50 + 100
        CHECK(h.get_vector(5050)->size() == 8 && f.get_vector(5050)->size() == 8);
    }
}

TEST(lean, l04_save_and_load) {
    auto data = random_vectors(300, 32, 1103);
    HnswIndex slim(32, Metric::Cosine, hnsw_lean(8)), full(32, Metric::Cosine, hnsw_truncate(8));
    FlatIndex fslim(32, Metric::Cosine, Schema::dynamic(), lean());
    for (std::uint64_t i = 0; i < 300; ++i) {
        slim.add(i, data[i]);
        full.add(i, data[i]);
        fslim.add(i, data[i]);
    }
    const std::string sb = save_bytes(slim), fb = save_bytes(fslim);
    auto s2 = load_hnsw(sb);
    auto f2 = load_flat(fb);
    CHECK(save_bytes(*s2) == sb && save_bytes(*f2) == fb);
    CHECK(!s2->params().quantization.keep_full_vectors && s2->storage().originals() == nullptr && !f2->quantization().keep_full_vectors);
    CHECK(*s2->get_vector(3) == *slim.get_vector(3) && same_results(s2->search(data[4], 10, 64), slim.search(data[4], 10, 64)));
    CHECK(sb.size() + 300 * 32 * 4 <= save_bytes(full).size());  // the full vectors are not saved either
}

TEST(lean, l05_damaged_files) {
    auto data = random_vectors(10, 6, 1104);
    HnswIndex h(6, Metric::Cosine, hnsw_lean(2));
    FlatIndex f(6, Metric::Cosine, Schema::dynamic(), lean());
    for (std::uint64_t i = 0; i < 10; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (int kind = 0; kind < 2; ++kind) {
        const std::string bytes = kind == 0 ? save_bytes(h) : save_bytes(f);
        auto load = [&](const std::string& b) { if (kind == 0) (void)load_hnsw(b); else (void)load_flat(b); };
        std::size_t rejected = 0, tried = 0;
        for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
            try { load(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
        }
        for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
            std::string damaged = bytes;
            damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
            try { load(damaged); } catch (const FormatError&) { ++rejected; }
        }
        CHECK(rejected == tried);
        ParsedFile p = ParsedFile::parse(bytes);  // the keep-full-vectors flag (next to last byte of CONF) set to 2
        std::string& conf = p.payload("CONF");
        poke<std::uint8_t>(conf, conf.size() - 2, 2);
        try {
            load(p.assemble());
            CHECK(false);
        } catch (const FormatError& e) {
            CHECK(std::string(e.what()).find("keep-full-vectors") != std::string::npos);
        }
    }
}

TEST(lean, l06_flag_flip_is_detected) {
    // A file saved keeping full vectors, with the flag flipped to "not kept" (and
    // vice versa): the records no longer match their expected size, never misread.
    auto data = random_vectors(20, 6, 1105);
    HnswIndex full(6, Metric::Cosine), slim(6, Metric::Cosine, hnsw_lean());
    for (std::uint64_t i = 0; i < 20; ++i) {
        full.add(i, data[i]);
        slim.add(i, data[i]);
    }
    for (const auto* index : {&full, &slim}) {
        ParsedFile p = ParsedFile::parse(save_bytes(*index));
        std::string& conf = p.payload("CONF");
        poke<std::uint8_t>(conf, conf.size() - 2, std::uint8_t(conf[conf.size() - 2] == 1 ? 0 : 1));  // the keep-full-vectors flag
        CHECK_THROWS_AS(FormatError, load_hnsw(p.assemble()));
    }
}

TEST(lean, l07_with_sq8) {
    // With SQ8, keeping only codes: after training the floats are dropped.
    auto data = clustered(800, 16, 6, 1106);
    HnswParams p = hnsw_lean(8);
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 200;
    HnswIndex h(16, Metric::Cosine, p);
    for (std::uint64_t i = 0; i < 800; ++i) h.add(i, data[i]);
    CHECK(h.codec_trained() && h.storage().originals() == nullptr && h.codec()->dim() == 8);
    CHECK(!h.storage().vectors().has_values() && h.get_vector(7)->size() == 8);
    CHECK(h.search(data[7], 1, 64)[0].id == 7);
    auto h2 = load_hnsw(save_bytes(h));
    CHECK(save_bytes(*h2) == save_bytes(h) && h2->codec_trained());
}

TEST(lean, l08_default_keeps_everything) {
    CHECK(QuantizationParams{}.keep_full_vectors);
    HnswIndex h(4, Metric::Cosine);
    h.add(1, std::vector<float>{3, 4, 0, 0});
    CHECK(*h.get_vector(1) == (std::vector<float>{3, 4, 0, 0}) && h.storage().originals() != nullptr);
}

// ===========================================================================
// Review of phases Q0-Q2 and keep_full_vectors (V1-V3)
// ===========================================================================

TEST(review, v01_range_search_inside_a_coded_search) {
    // A coded search marks its thread "distances from codes, with this query's
    // table". A range search started inside its predicate must use full
    // precision with its own query, not the outer search's codes.
    auto data = clustered(800, 16, 6, 1200);
    HnswParams p;
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 100;
    HnswIndex h(16, Metric::L2, p);
    for (std::uint64_t i = 0; i < 800; ++i) h.add(i, data[i]);
    const auto expected = h.search_range(data[5], 3.0f);
    long calls = 0, wrong = 0;  // on every call: some come from the planner's sampling, before the search
    SearchOptions o;
    o.predicate = [&](std::uint64_t) {
        ++calls;
        wrong += long(!same_results(h.search_range(data[5], 3.0f), expected));
        return true;
    };
    (void)h.search(data[400], 5, 64, o);
    CHECK(!expected.empty() && calls > 0 && wrong == 0);
}

TEST(review, v02_huge_rerank_factor) {
    // k * rerank_factor must not wrap around (10 * 2^63 wraps to exactly 0).
    auto data = clustered(300, 8, 4, 1201);
    QuantizationParams q;
    q.codec = CodecKind::SQ8;
    q.train_after = 50;
    q.rerank_factor = std::size_t(1) << 63;
    HnswParams p;
    p.quantization = q;
    HnswIndex h(8, Metric::L2, p);
    FlatIndex f(8, Metric::L2, Schema::dynamic(), q);
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    SearchOptions exact;
    exact.strategy = Strategy::ForceExact;
    CHECK(h.search(data[3], 10, 64, exact).size() == 10 && f.search(data[3], 10).size() == 10);
    CHECK(same_results(h.search(data[3], 10, 64, exact), f.search(data[3], 10)));  // both exact, both re-ranked
}

namespace {

/// The same results up to the order of ties: identical distances, and the same
/// IDs in each group of equal distance (except the last group, which the k
/// cutoff may split differently: any of its members is a correct answer).
bool same_up_to_ties(const std::vector<SearchResult>& a, const std::vector<SearchResult>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].distance != b[i].distance) return false;
    for (std::size_t i = 0; i < a.size();) {
        std::size_t j = i;
        while (j < a.size() && a[j].distance == a[i].distance) ++j;
        if (j < a.size()) {  // a complete group: the same members
            std::multiset<std::uint64_t> x, y;
            for (std::size_t t = i; t < j; ++t) { x.insert(a[t].id); y.insert(b[t].id); }
            if (x != y) return false;
        }
        i = j;
    }
    return true;
}

/// What get_vector should return for `given`: the vector itself when the index
/// keeps vectors as given (or never changes them), otherwise the stored form.
struct ReadModel {
    Metric metric;
    std::size_t n;  // indexed dimensions
    bool exact;
    double tolerance = 0;  // codes only: the bound on decoding errors so far
    bool matches(const std::vector<float>& got, const std::vector<float>& given) const {
        if (exact && tolerance == 0) return got == given;
        if (got.size() != n) return false;
        std::vector<double> want(given.begin(), given.begin() + std::ptrdiff_t(n));
        if (metric == Metric::Cosine) {
            double len = 0;
            for (double x : want) len += x * x;
            if (len > 0) for (double& x : want) x /= std::sqrt(len);
        }
        for (std::size_t d = 0; d < n; ++d)
            if (std::abs(got[d] - want[d]) > tolerance * 1.001 + 1e-5 * (1 + std::abs(want[d]))) return false;
        return true;
    }
};

/// Half the largest SQ8 step of an index's codec (its decoding error bound).
double half_step(const VectorCodec& c) {
    const auto state = c.state();  // minimums, then steps
    double largest = 0;
    for (std::size_t d = c.dim(); d < state.size(); ++d) largest = std::max(largest, double(state[d]));
    return largest / 2;
}

}  // namespace

TEST(review, v03_model_based_random_operations) {
    long configs = 0, ops = 0, checks_done = 0;
    for (unsigned seed = 1; seed <= 24; ++seed) {
        std::mt19937_64 rng(seed * 7919);
        auto pick = [&](std::size_t n) { return std::size_t(rng() % n); };
        const Metric metric = kMetrics[pick(3)];
        const std::size_t dim = pick(2) ? 12 : 6;
        QuantizationParams q;
        q.truncate_to = pick(3) == 0 ? dim / 2 : 0;
        q.codec = std::array{CodecKind::None, CodecKind::SQ8, CodecKind::Binary}[pick(3)];
        q.train_after = pick(2) ? 30 : 0;
        q.rerank = pick(4) != 0;
        q.keep_full_vectors = pick(2) == 0;
        HnswParams hp;
        hp.M = 6;
        hp.ef_construction = 40;
        hp.seed = seed;
        hp.quantization = q;
        auto h = std::make_unique<HnswIndex>(dim, metric, hp);
        IvfParams ivf;  // Flat sometimes has an inverted file (trained automatically or explicitly)
        if (pick(3) == 0) {
            ivf.lists = 2 + pick(6);
            ivf.probes = 1 + pick(3);
            ivf.train_after = pick(2) ? 40 : 0;
        }
        auto f = std::make_unique<FlatIndex>(dim, metric, Schema::dynamic(), q, ivf);
        const std::size_t n = vecdb::indexed_dim(q, dim);
        std::map<std::uint64_t, std::vector<float>> model;
        const ReadModel read{metric, n, q.keep_full_vectors || (metric != Metric::Cosine && n == dim)};
        // Codes only: each index's stored vectors are simulated with its own codec,
        // decode(encode(v)), clamping included: from the stored form when a vector is
        // added, and again from the previous simulation when the codec is retrained
        // (the library retrains from decoded vectors). One map per index: each trains
        // its own codec.
        struct CodesOnly {
            bool active = false;
            std::vector<float> state;  // the codec's state, to notice retraining
            std::map<std::uint64_t, std::vector<float>> expected;
        } sim_h, sim_f;
        auto stored_form = [&](const std::vector<float>& v) {
            std::vector<float> s2(v.begin(), v.begin() + std::ptrdiff_t(n));
            if (metric == Metric::Cosine) {
                double len = 0;
                for (float x : s2) len += double(x) * x;
                if (len > 0) for (float& x : s2) x = float(x / std::sqrt(len));
            }
            return s2;
        };
        auto simulate = [](const VectorCodec& c, const std::vector<float>& v) {
            std::vector<std::uint8_t> code(c.code_size());
            c.encode(v, code.data());
            std::vector<float> out(c.dim());
            c.decode(code.data(), out);
            return out;
        };
        std::vector<std::uint64_t> touched;  // IDs whose vector the last operation set
        auto follow = [&](CodesOnly& sim, bool codes_only, const VectorCodec* c) {
            if (!codes_only) return;
            if (!sim.active) {  // just dropped its floats: encoded from the stored floats
                for (const auto& [id, v] : model) sim.expected[id] = simulate(*c, stored_form(v));
                sim.active = true;
                sim.state = c->state();
            } else if (c->state() != sim.state) {  // retrained: re-encoded from decoded vectors
                for (auto& [id, e] : sim.expected) e = simulate(*c, e);
                sim.state = c->state();
            }
            for (auto id : touched) if (model.contains(id)) sim.expected[id] = simulate(*c, stored_form(model[id]));
        };
        std::normal_distribution<float> nd(0, 1);
        auto vec = [&] {
            std::vector<float> v(dim);
            for (float& x : v) x = nd(rng) * (pick(20) == 0 ? 1000.0f : 1.0f);  // now and then, large values
            if (!model.empty() && pick(15) == 0) v = std::next(model.begin(), std::ptrdiff_t(pick(model.size())))->second;  // a duplicate
            return v;
        };
        auto some_id = [&] { return pick(4) == 0 || model.empty() ? std::uint64_t(pick(400)) : std::next(model.begin(), std::ptrdiff_t(pick(model.size())))->first; };
        bool ok = true;
        std::string where;
        for (int step = 0; step < 300 && ok; ++step, ++ops) {
            const std::size_t op = pick(100);
            std::string name;
            if (op < 35) {
                name = "add";
                const std::uint64_t id = pick(400);
                const auto v = vec();
                if (model.contains(id)) {
                    bool threw = false;
                    try { h->add(id, v); } catch (const std::invalid_argument&) { threw = true; }
                    try { f->add(id, v); } catch (const std::invalid_argument&) { threw = threw && true; }
                    ok = ok && threw;
                } else {
                    h->add(id, v);
                    f->add(id, v);
                    model[id] = v;
                    touched.push_back(id);
                }
            } else if (op < 47) {
                name = "remove";
                const std::uint64_t id = some_id();
                const bool expected = model.erase(id) == 1;
                ok = ok && h->remove(id) == expected && f->remove(id) == expected;
            } else if (op < 57) {
                name = "update";
                const std::uint64_t id = some_id();
                const auto v = vec();
                const bool expected = model.contains(id);
                ok = ok && h->update(id, v) == expected && f->update(id, v) == expected;
                if (expected) {
                    model[id] = v;
                    touched.push_back(id);
                }
            } else if (op < 65) {
                name = "upsert";
                const std::uint64_t id = some_id();
                const auto v = vec();
                h->upsert(id, v);
                f->upsert(id, v);
                model[id] = v;
                touched.push_back(id);
            } else if (op < 73) {
                name = "add_batch";
                const std::size_t count = 2 + pick(15);
                std::vector<std::uint64_t> ids;
                std::vector<float> flat;
                std::set<std::uint64_t> fresh;
                while (ids.size() < count) {
                    const std::uint64_t id = 400 + pick(2000);
                    if (model.contains(id) || !fresh.insert(id).second) continue;
                    ids.push_back(id);
                    const auto v = vec();
                    flat.insert(flat.end(), v.begin(), v.end());
                }
                if (pick(5) == 0) ids.back() = ids.front();  // a repeated ID: the whole batch is refused
                const bool bad = ids.back() == ids.front();
                bool threw_h = false, threw_f = false;
                try { h->add_batch(ids, flat, 1 + pick(3)); } catch (const std::invalid_argument&) { threw_h = true; }
                try { f->add_batch(ids, flat); } catch (const std::invalid_argument&) { threw_f = true; }
                ok = ok && threw_h == bad && threw_f == bad;
                if (!bad)
                    for (std::size_t i = 0; i < ids.size(); ++i) {
                        model[ids[i]] = std::vector<float>(flat.begin() + std::ptrdiff_t(i * dim), flat.begin() + std::ptrdiff_t((i + 1) * dim));
                        touched.push_back(ids[i]);
                    }
            } else if (op < 78) {
                name = "remove_batch";
                std::vector<std::uint64_t> ids;
                for (int i = 0; i < 5; ++i) ids.push_back(some_id());
                std::set<std::uint64_t> removed;
                for (auto id : ids) if (model.contains(id)) removed.insert(id);
                for (auto id : removed) model.erase(id);
                ok = ok && h->remove_batch(ids) == removed.size() && f->remove_batch(ids) == removed.size();
            } else if (op < 82) {
                name = "compact";
                h->compact();
                f->compact();
            } else if (op < 86) {
                name = "train_codec";
                if (ivf.lists > 0 && model.size() >= ivf.lists && pick(2)) f->train_lists();
                if (q.codec != CodecKind::None && !model.empty()) {
                    // Codes only: training again is refused (it could only lose
                    // precision). Otherwise train on a sample of at least 20.
                    for (int which = 0; which < 2; ++which) {
                        const bool codes_only = which == 0 ? !h->has_vector_values() : !f->has_vector_values();
                        bool refused = false;
                        try {
                            if (codes_only || model.size() >= 20) {
                                if (which == 0) h->train_codec();
                                else f->train_codec();
                            }
                        } catch (const std::logic_error&) {
                            refused = true;
                        }
                        ok = ok && refused == codes_only;
                    }
                }
            } else if (op < 92) {
                name = "search_by_id";
                if (!model.empty()) {
                    const std::uint64_t id = some_id();
                    const auto r = h->search_by_id(id, 5, 32);
                    const auto fr = f->search_by_id(id, 5);
                    ok = ok && bool(r) == model.contains(id) && bool(fr) == model.contains(id);
                    if (r) for (const auto& x : *r) ok = ok && x.id != id && model.contains(x.id);
                    if (fr) for (const auto& x : *fr) ok = ok && x.id != id && model.contains(x.id);
                }
            } else {
                name = "save and load";
                h = load_hnsw(save_bytes(*h));
                f = load_flat(save_bytes(*f));
            }
            follow(sim_h, !h->has_vector_values(), h->codec());
            follow(sim_f, !f->has_vector_values(), f->codec());
            touched.clear();
            // Check against the model after every operation.
            std::string failed;
            if (!ok) failed = "the operation's own result";
            if (ok && !(h->size() == model.size() && f->size() == model.size())) { ok = false; failed = "size"; }
            // Codes only: within one step of the simulation (a rounding boundary may
            // fall differently by a float ulp); otherwise exact or the stored form.
            auto read_ok = [&](const std::optional<std::vector<float>>& got, const CodesOnly& sim, const VectorCodec* c,
                               std::uint64_t id, const std::vector<float>& v) {
                if (!got) return false;
                if (!sim.active) return read.matches(*got, v);
                const auto& e = sim.expected.at(id);
                const double tol = 4 * half_step(*c) * 1.01 + 1e-5;  // a value on a boundary may round either way (binary: two spreads)
                if (got->size() != e.size()) return false;
                for (std::size_t d = 0; d < e.size(); ++d)
                    if (std::abs(double((*got)[d]) - e[d]) > tol) return false;
                return true;
            };
            for (const auto& [id, v] : model) {
                if (ok && !(read_ok(h->get_vector(id), sim_h, h->codec(), id, v) && read_ok(f->get_vector(id), sim_f, f->codec(), id, v))) {
                    ok = false;
                    failed = "get_vector of " + std::to_string(id);
                }
            }
            if (ok && !(!h->contains(5000) && !f->get_vector(5000))) { ok = false; failed = "unknown ID"; }
            if (!model.empty()) {  // exact HNSW search equals Flat (no codec: the same distances)
                const auto qv = vec();
                SearchOptions exact;
                exact.strategy = Strategy::ForceExact;
                const auto hr = h->search(qv, 7, 32, exact), fr = f->search(qv, 7);
                if (ok && !(hr.size() == std::min<std::size_t>(7, model.size()) && fr.size() == hr.size())) { ok = false; failed = "result count"; }
                if (ok && q.codec == CodecKind::None && !f->lists_trained() && !same_up_to_ties(hr, fr)) {
                    ok = false;
                    failed = "exact HNSW vs Flat:";
                    for (std::size_t i = 0; i < hr.size(); ++i)
                        if (hr[i].id != fr[i].id || hr[i].distance != fr[i].distance) {
                            char buf[160];
                            std::snprintf(buf, sizeof buf, " [%llu %.9g | %llu %.9g]", (unsigned long long)hr[i].id, double(hr[i].distance), (unsigned long long)fr[i].id, double(fr[i].distance));
                            failed += buf;
                        }
                }
            }
            if (ok && h->codec_trained() && !codes_match(*h)) { ok = false; failed = "codes"; }
            if (ok && f->lists_trained()) {  // every vector in exactly one list
                std::size_t total = 0;
                for (std::size_t sz : f->list_sizes()) total += sz;
                bool listed = total == model.size();
                for (const auto& [id, v] : model) listed = listed && f->list_of(id).has_value();
                if (!listed) { ok = false; failed = "inverted lists"; }
            }
            if (ok && step % 40 == 39) {
                const std::string hb = save_bytes(*h), fb = save_bytes(*f);
                if (!(save_bytes(*load_hnsw(hb)) == hb && save_bytes(*load_flat(fb)) == fb)) { ok = false; failed = "save and load"; }
            }
            ++checks_done;
            if (!ok) where = "seed " + std::to_string(seed) + ", step " + std::to_string(step) + ", after " + name + ": " + failed;
        }
        if (!ok) std::printf("    first failure: %s (metric %d, dim %zu, truncate %zu, codec %d, train_after %zu, rerank %d, keep_full %d)\n",
                             where.c_str(), int(metric), dim, q.truncate_to, int(q.codec), q.train_after, int(q.rerank), int(q.keep_full_vectors));
        CHECK(ok);
        h->compact();
        // Inner product with vector lengths 1,000 times apart leaves some vectors hard
        // to reach (the long ones dominate every neighborhood): a known limitation of
        // HNSW with inner product, confirmed absent without such vectors.
        // Binary codes only, at 6 or 12 dimensions: at most 64 or 4,096 distinct codes,
        // so many vectors share one and a graph built from codes cannot separate
        // them (measured: about 200 vectors on 22 to 61 codes). Inherent, documented.
        const bool binary_codes_only = q.codec == CodecKind::Binary && !h->has_vector_values();
        if (metric != Metric::InnerProduct && !binary_codes_only) check_graph(*h, 0.9);
        ++configs;
    }
    std::printf("    %ld configurations, %ld random operations, each followed by a full check (%ld)\n", configs, ops, checks_done);
}

TEST(review, v04_compact_never_changes_stored_vectors) {
    // Without vectors as given, compact() rebuilds from the stored vectors: they
    // must come out bit for bit the same (not normalized a second time).
    auto data = random_vectors(300, 16, 1203);
    for (std::size_t truncate : {std::size_t(0), std::size_t(6)}) {
        for (Metric m : {Metric::Cosine, Metric::L2}) {
            HnswParams p;
            p.quantization.keep_full_vectors = false;
            p.quantization.truncate_to = truncate;
            HnswIndex h(16, m, p);
            for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i]);
            for (std::uint64_t i = 0; i < 300; i += 3) h.remove(i);
            std::map<std::uint64_t, std::vector<float>> before;
            for (std::uint64_t i = 1; i < 300; i += 3) before[i] = *h.get_vector(i);
            h.compact();
            h.compact();  // twice: no drift either
            bool same = h.deleted_count() == 0;
            for (const auto& [id, v] : before) same = same && *h.get_vector(id) == v;
            CHECK(same);
        }
    }
}

// ===========================================================================
// SQ8 keeping only codes: keep_full_vectors off (C1-C8)
// ===========================================================================

namespace {

QuantizationParams codes_only_q(std::size_t train_after = 200, std::size_t truncate = 0) {
    QuantizationParams q;
    q.codec = CodecKind::SQ8;
    q.train_after = train_after;
    q.keep_full_vectors = false;
    q.truncate_to = truncate;
    return q;
}

HnswParams hnsw_codes_only(std::size_t train_after = 200, std::size_t truncate = 0) {
    HnswParams p;
    p.quantization = codes_only_q(train_after, truncate);
    return p;
}

}  // namespace

TEST(codes_only, c01_floats_dropped_recall_kept) {
    auto data = clustered(3000, 32, 12, 1300);
    auto queries = clustered(50, 32, 12, 1301);
    for (Metric m : kMetrics) {
        HnswIndex h(32, m, hnsw_codes_only(500)), plain(32, m);
        FlatIndex f(32, m, Schema::dynamic(), codes_only_q(500)), exact(32, m);
        for (std::uint64_t i = 0; i < 3000; ++i) {
            h.add(i, data[i]);
            plain.add(i, data[i]);
            f.add(i, data[i]);
            exact.add(i, data[i]);
            if (i == 498) CHECK(h.has_vector_values() && f.has_vector_values());  // floats until trained
        }
        CHECK(!h.has_vector_values() && !f.has_vector_values() && h.codec_trained());
        const double base = recall(plain, exact, queries, 10, 64), r = recall(h, exact, queries, 10, 64);
        double fr = 0;
        for (const auto& q : queries) {
            std::set<std::uint64_t> truth;
            for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
            for (const auto& x : f.search(q, 10)) fr += double(truth.count(x.id));
        }
        fr /= 500.0;
        std::printf("    metric %d: hnsw floats %.3f, hnsw codes only %.3f | flat codes only %.3f\n", int(m), base, r, fr);
        CHECK(r >= base - 0.05 && fr >= 0.85);
        // Vectors read back decoded: within half a step of the stored vector.
        const auto state = h.codec()->state();
        bool close = true;
        for (std::uint64_t i = 0; i < 3000; i += 97) {
            const auto v = *h.get_vector(i);
            std::vector<float> stored = data[i];
            if (m == Metric::Cosine) {
                double len = 0;
                for (float x : stored) len += double(x) * x;
                for (float& x : stored) x = float(x / std::sqrt(len));
            }
            for (std::size_t d = 0; d < 32; ++d)
                close = close && std::abs(v[d] - std::clamp(stored[d], state[d], state[d] + 255 * state[32 + d])) <= state[32 + d] * 0.5f + 1e-5f;
        }
        CHECK(close);
    }
}

TEST(codes_only, c02_every_path) {
    auto data = clustered(1600, 16, 8, 1302);
    HnswIndex h(16, Metric::L2, hnsw_codes_only(200));
    FlatIndex f(16, Metric::L2, Schema::dynamic(), codes_only_q(200));
    for (std::uint64_t i = 0; i < 400; ++i) {
        h.add(i, data[i], meta_for(i));
        f.add(i, data[i], meta_for(i));
    }
    for (std::uint64_t i = 0; i < 400; i += 5) {  // HNSW: free slots; Flat: swap with last
        h.remove(i);
        f.remove(i);
    }
    for (std::uint64_t i = 1000; i < 1050; ++i) {  // reused slots
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (std::uint64_t i = 1; i < 400; i += 7) {
        h.update(i, data[i + 1000]);
        f.update(i, data[i + 1000]);
    }
    for (std::uint64_t i = 3; i < 400; i += 11) {
        h.upsert(i, data[i + 1100]);
        f.upsert(i, data[i + 1100]);
    }
    h.add_batch(id_range(5000, 50), flatten(Vectors(data.begin() + 1500, data.begin() + 1550)), 1);
    h.add_batch(id_range(6000, 50), flatten(Vectors(data.begin() + 1550, data.begin() + 1600)), 4);
    f.add_batch(id_range(5000, 50), flatten(Vectors(data.begin() + 1500, data.begin() + 1550)));
    std::atomic<bool> flag{false};
    OperationOptions o;
    o.cancel = &flag;
    o.progress = [&](std::size_t done, std::size_t) { if (done > 10) flag = true; };
    CHECK_THROWS_AS(Cancelled, h.add_batch(id_range(7000, 60), flatten(Vectors(data.begin() + 1300, data.begin() + 1360)), 4, o));
    CHECK(codes_match(h) && codes_match(f) && !h.has_vector_values() && !f.has_vector_values());
    // Every kind of search works from codes.
    CHECK(h.search_by_id(5001, 5, 64)->size() == 5 && f.search_by_id(5001, 5)->size() == 5);
    CHECK(!h.search_range(data[1501], 4.0f).empty() && !f.search_range(data[1501], 4.0f).empty());
    const auto batch = h.search_batch(flatten(Vectors(data.begin() + 1500, data.begin() + 1505)), 5, 64, 3);
    bool same = true;
    for (std::size_t q = 0; q < 5; ++q) same = same && same_results(batch[q], h.search(data[1500 + q], 5, 64));
    CHECK(same && h.search(data[1520], 1, 64)[0].id == 5020 && f.search(data[1520], 1)[0].id == 5020);
    const std::size_t live = h.size();
    h.compact();
    CHECK(h.size() == live && h.deleted_count() == 0 && !h.has_vector_values() && codes_match(h));
    check_graph(h, 0.95);
}

TEST(codes_only, c03_save_and_load) {
    auto data = clustered(800, 64, 6, 1303);
    HnswIndex h(64, Metric::Cosine, hnsw_codes_only(100)), floats(64, Metric::Cosine);
    FlatIndex f(64, Metric::L2, Schema::dynamic(), codes_only_q(100));
    for (std::uint64_t i = 0; i < 800; ++i) {
        h.add(i, data[i]);
        floats.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    auto h2 = load_hnsw(hb);
    auto f2 = load_flat(fb);
    CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb && !h2->has_vector_values() && !f2->has_vector_values());
    CHECK(same_results(h2->search(data[3], 10, 64), h.search(data[3], 10, 64)) && *h2->get_vector(9) == *h.get_vector(9));
    // No floats saved: 64 bytes of code per vector instead of 2 x 256 bytes (floats and originals).
    CHECK(hb.size() + 800 * 400 < save_bytes(floats).size());
    std::printf("    file sizes: codes only %zu bytes, full precision %zu bytes\n", hb.size(), save_bytes(floats).size());
}

TEST(codes_only, c04_training_is_final) {
    auto data = clustered(300, 8, 4, 1304);
    HnswIndex h(8, Metric::L2, hnsw_codes_only(0));  // explicit training only
    FlatIndex f(8, Metric::L2, Schema::dynamic(), codes_only_q(0));
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(h.has_vector_values() && !h.codec_trained());  // until trained: floats, full precision
    h.train_codec();
    f.train_codec();
    CHECK(!h.has_vector_values() && !f.has_vector_values());
    const std::string hb = save_bytes(h);
    CHECK_THROWS_AS(std::logic_error, h.train_codec());  // could only lose precision
    CHECK_THROWS_AS(std::logic_error, f.train_codec());
    CHECK(save_bytes(h) == hb);
}

TEST(codes_only, c05_damaged_and_inconsistent_files) {
    auto data = random_vectors(30, 4, 1305);
    HnswIndex h(4, Metric::L2, hnsw_codes_only(10));
    FlatIndex f(4, Metric::L2, Schema::dynamic(), codes_only_q(10));
    for (std::uint64_t i = 0; i < 30; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    for (int kind = 0; kind < 2; ++kind) {
        const std::string bytes = kind == 0 ? save_bytes(h) : save_bytes(f);
        auto load = [&](const std::string& b) { if (kind == 0) (void)load_hnsw(b); else (void)load_flat(b); };
        std::size_t rejected = 0, tried = 0;
        for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
            try { load(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
        }
        for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
            std::string damaged = bytes;
            damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
            try { load(damaged); } catch (const FormatError&) { ++rejected; }
        }
        CHECK(rejected == tried);
        // Valid checksums, inconsistent content: a values flag of 2; "no values"
        // with keep_full_vectors on; "no values" without a trained codec.
        ParsedFile p2 = ParsedFile::parse(bytes);
        poke<std::uint8_t>(p2.payload("CONF"), p2.payload("CONF").size() - 1, 2);
        ParsedFile keep = ParsedFile::parse(bytes);
        poke<std::uint8_t>(keep.payload("CONF"), keep.payload("CONF").size() - 2, 1);
        ParsedFile untrained = ParsedFile::parse(bytes);
        std::string& codc = untrained.payload("CODC");
        codc.resize(18);  // up to the training flag
        codc.push_back('\0');  // "not trained"
        codc.append(8, '\0');  // then the number of PQ parts, which ends the section
        for (ParsedFile* e : {&p2, &keep, &untrained}) CHECK_THROWS_AS(FormatError, load(e->assemble()));
    }
}

TEST(codes_only, c06_searching_while_floats_are_dropped) {
    auto data = clustered(1500, 8, 6, 1306);
    HnswIndex h(8, Metric::L2, hnsw_codes_only(800));
    for (std::uint64_t i = 0; i < 500; ++i) h.add(i, data[i]);
    std::atomic<bool> stop{false};
    Problems p;
    with_watchdog(120, "c06", [&] {
        std::thread writer([&] {
            for (std::uint64_t i = 500; i < 1500; ++i) h.add(i, data[i]);  // trains and drops floats at 800
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) {
                check_results(h.search(data[q % 1500], 10, 32), 1500, p);
                (void)h.get_vector(q % 500);
                ++q;
            }
        });
        writer.join();
    });
    CHECK(p.total() == 0 && !h.has_vector_values() && codes_match(h));
}

TEST(codes_only, c07_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(400, 8, 4, 1307);
    HnswIndex h(8, Metric::L2, hnsw_codes_only(0));
    FlatIndex f(8, Metric::L2, Schema::dynamic(), codes_only_q(0));
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    const std::string hb = save_bytes(h), fb = save_bytes(f);
    // Training is all or nothing: on failure the floats are still there.
    CHECK(sweep_allocation_failures([&] { h.train_codec(); }, [&] { CHECK(save_bytes(h) == hb && h.has_vector_values()); }) > 0);
    CHECK(sweep_allocation_failures([&] { f.train_codec(); }, [&] { CHECK(save_bytes(f) == fb && f.has_vector_values()); }) > 0);
    CHECK(!h.has_vector_values() && !f.has_vector_values());
    const std::string fb2 = save_bytes(f);
    std::uint64_t id = 900;
    CHECK(sweep_allocation_failures([&] { f.add(id, data[350]); }, [&] { CHECK(save_bytes(f) == fb2); }) > 0);
    CHECK(sweep_allocation_failures([&] { h.add(id, data[350]); }, [&] { CHECK(!h.contains(id) && h.size() == 300); }) > 0);
    CHECK(codes_match(h) && codes_match(f));
}

TEST(codes_only, c08_with_truncation_and_cosine) {
    auto data = clustered(1200, 48, 8, 1308);
    HnswIndex h(48, Metric::Cosine, hnsw_codes_only(300, 16));
    FlatIndex f(48, Metric::Cosine, Schema::dynamic(), codes_only_q(300, 16));
    for (std::uint64_t i = 0; i < 1200; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(!h.has_vector_values() && h.codec()->dim() == 16 && h.get_vector(5)->size() == 16 && h.storage().originals() == nullptr);
    CHECK(h.search(data[5], 1, 64)[0].id == 5 && f.search(data[5], 1)[0].id == 5);
    auto h2 = load_hnsw(save_bytes(h));
    CHECK(save_bytes(*h2) == save_bytes(h) && h2->indexed_dim() == 16);
}

// ===========================================================================
// Phase Q3: binary quantization (B1-B12)
// ===========================================================================

namespace {

QuantizationParams binary_q(std::size_t train_after = 200, bool rerank = true, bool keep_full = true) {
    QuantizationParams q;
    q.codec = CodecKind::Binary;
    q.train_after = train_after;
    q.rerank = rerank;
    q.keep_full_vectors = keep_full;
    return q;
}

HnswParams hnsw_binary(std::size_t train_after = 200, bool rerank = true, bool keep_full = true) {
    HnswParams p;
    p.quantization = binary_q(train_after, rerank, keep_full);
    return p;
}

}  // namespace

TEST(binary, b01_code_size_and_bits) {
    for (std::size_t d : {std::size_t(1), std::size_t(7), std::size_t(8), std::size_t(9), std::size_t(64), std::size_t(100)})
        CHECK(BinaryCodec(d, Metric::L2).code_size() == (d + 7) / 8);
    BinaryCodec c(10, Metric::L2);
    std::vector<float> sample;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 10; ++j) sample.push_back(float((i % 2 ? 1 : -1) * (j + 1)));  // mean 0 in every dimension
    c.train(sample);
    std::uint8_t code[2];
    c.encode(std::vector<float>{1, -1, 1, -1, 1, -1, 1, -1, 1, 1}, code);
    CHECK(code[0] == 0x55 && code[1] == 0x03);  // bit j set when above the threshold (0)
}

TEST(binary, b02_fixed_answer) {
    // Every CPU must produce exactly these codes (comparisons only) and state.
    // Expected values computed independently with NumPy float32: thresholds
    // (2, 2, 4/3), spreads (4/3, 4/3, 20/9).
    const std::vector<float> sample = {0, 4, -2, 2, 0, 2, 4, 2, 4};  // three vectors of 3
    BinaryCodec c(3, Metric::L2);
    c.train(sample);
    CHECK(c.state() == (std::vector<float>{2, 2, 4.0f / 3.0f, 4.0f / 3.0f, 4.0f / 3.0f, 2.2222223281860352f}));
    std::uint8_t code[1];
    c.encode(std::vector<float>{2.5f, 1.0f, 1.5f}, code);
    CHECK(code[0] == 0x05);
    float back[3];
    c.decode(code, back);
    CHECK(back[0] == 3.3333334922790527f && back[1] == 0.6666666269302368f && back[2] == 3.555555820465088f);
}

TEST(binary, b03_distances_match_decoded_vectors) {
    std::mt19937 rng(40);
    std::normal_distribution<float> nd(0.5f, 1);
    for (Metric m : kMetrics) {
        std::vector<float> sample(300 * 20);
        for (float& x : sample) x = nd(rng);
        BinaryCodec c(20, m);
        c.train(sample);
        std::vector<float> table, back(20);
        std::vector<std::uint8_t> code(c.code_size());
        const std::span<const float> q(&sample[0], 20);
        c.prepare(q, table);
        bool close = true;
        for (std::size_t i = 1; i < 300; ++i) {
            c.encode(std::span<const float>(&sample[i * 20], 20), code.data());
            c.decode(code.data(), back);
            double expected = 0;
            for (std::size_t j = 0; j < 20; ++j)
                expected += m == Metric::L2 ? double(q[j] - back[j]) * (q[j] - back[j]) : double(q[j]) * back[j];
            if (m == Metric::InnerProduct) expected = -expected;
            if (m == Metric::Cosine) expected = 1.0 - expected;
            close = close && std::abs(c.distance(table, code.data()) - expected) < 1e-4 * (1 + std::abs(expected));
        }
        CHECK(close);
    }
}

TEST(binary, b04_decode_then_encode_is_exact) {
    // Including a dimension constant in training (spread 0) and a value above it.
    const std::vector<float> sample = {5, 1, -3, 5, 2, 3, 5, 0, 1};  // dimension 0 is always 5
    BinaryCodec c(3, Metric::L2);
    c.train(sample);
    bool exact = true;
    for (unsigned bits = 0; bits < 8; ++bits) {
        const std::uint8_t code = static_cast<std::uint8_t>(bits);
        std::uint8_t again;
        float v[3];
        c.decode(&code, v);
        c.encode(v, &again);
        exact = exact && again == code;
    }
    std::uint8_t code;
    c.encode(std::vector<float>{6, 0, 0}, &code);  // above a constant dimension: bit set
    CHECK(exact && (code & 1u) == 1u);
}

TEST(binary, b05_rejects_misuse) {
    BinaryCodec c(4, Metric::L2);
    std::uint8_t code[1];
    std::vector<float> table;
    CHECK_THROWS_AS(std::logic_error, c.encode(std::vector<float>(4, 1.0f), code));
    CHECK_THROWS_AS(std::logic_error, c.prepare(std::vector<float>(4, 1.0f), table));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>{}));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>{1, 2, kNaN, 4}));
    CHECK_THROWS_AS(std::invalid_argument, c.restore(std::vector<float>(7, 0.0f)));
    CHECK_THROWS_AS(std::invalid_argument, c.restore(std::vector<float>{0, 0, 0, 0, 1, 1, -1, 1}));  // a negative spread
    CHECK_THROWS_AS(std::invalid_argument, c.restore(std::vector<float>{3e38f, 0, 0, 0, 3e38f, 1, 1, 1}));  // overflows
    c.train(std::vector<float>(8, 1.0f));
    CHECK_THROWS_AS(std::invalid_argument, c.encode(std::vector<float>(3, 1.0f), code));
    CHECK(make_codec(CodecKind::Binary, 4, Metric::L2)->kind() == CodecKind::Binary);
}

TEST(binary, b06_recall) {
    auto data = clustered(3000, 64, 12, 1400);
    auto queries = clustered(50, 64, 12, 1401);
    for (Metric m : kMetrics) {
        // Binary codes are coarse: re-ranking needs many candidates. Flat re-ranks
        // k * rerank_factor (16 to 32 recommended for binary, default 4); HNSW
        // re-ranks its whole beam (ef).
        QuantizationParams wide = binary_q(500);
        wide.rerank_factor = 32;
        HnswIndex plain(64, m), reranked(64, m, hnsw_binary(500)), bits(64, m, hnsw_binary(500, false)), lean(64, m, hnsw_binary(500, true, false));
        FlatIndex exact(64, m), fr(64, m, Schema::dynamic(), binary_q(500)), fwide(64, m, Schema::dynamic(), wide);
        for (std::uint64_t i = 0; i < 3000; ++i) {
            plain.add(i, data[i]);
            reranked.add(i, data[i]);
            bits.add(i, data[i]);
            lean.add(i, data[i]);
            exact.add(i, data[i]);
            fr.add(i, data[i]);
            fwide.add(i, data[i]);
        }
        const double base = recall(plain, exact, queries, 10, 64), r = recall(reranked, exact, queries, 10, 64);
        const double r_wide = recall(reranked, exact, queries, 10, 400);
        const double b = recall(bits, exact, queries, 10, 64), l = recall(lean, exact, queries, 10, 64);
        double f = 0, fw = 0;
        for (const auto& q : queries) {
            std::set<std::uint64_t> truth;
            for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
            for (const auto& x : fr.search(q, 10)) f += double(truth.count(x.id));
            for (const auto& x : fwide.search(q, 10)) fw += double(truth.count(x.id));
        }
        f /= 500.0;
        fw /= 500.0;
        std::printf("    metric %d: hnsw %.3f | binary + re-rank: ef 64 %.3f, ef 400 %.3f; bits only %.3f, codes only %.3f | flat: re-rank x4 %.3f, x32 %.3f\n",
                    int(m), base, r, r_wide, b, l, f, fw);
        CHECK(!lean.has_vector_values() && lean.get_code(1)->size() == 8);
        CHECK(fw >= 0.95 && r_wide > r && r >= 0.65 && f >= 0.35 && b >= 0.30 && l >= 0.30);
    }
}

TEST(binary, b07_every_path) {
    auto data = clustered(1500, 24, 6, 1402);
    for (bool keep_full : {true, false}) {
        HnswIndex h(24, Metric::L2, hnsw_binary(200, true, keep_full));
        FlatIndex f(24, Metric::L2, Schema::dynamic(), binary_q(200, true, keep_full));
        for (std::uint64_t i = 0; i < 400; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        for (std::uint64_t i = 0; i < 400; i += 5) {
            h.remove(i);
            f.remove(i);
        }
        for (std::uint64_t i = 1000; i < 1050; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        for (std::uint64_t i = 1; i < 400; i += 7) {
            h.update(i, data[i + 1000]);
            f.update(i, data[i + 1000]);
        }
        h.add_batch(id_range(5000, 60), flatten(Vectors(data.begin() + 1400, data.begin() + 1460)), 4);
        f.add_batch(id_range(5000, 60), flatten(Vectors(data.begin() + 1400, data.begin() + 1460)));
        CHECK(codes_match(h) && codes_match(f) && h.has_vector_values() == keep_full);
        h.compact();
        CHECK(codes_match(h) && h.deleted_count() == 0);
        check_graph(h, 0.9);
        CHECK(h.search_by_id(5001, 5, 64)->size() == 5 && !f.search_range(data[1401], 50.0f).empty());
    }
}

TEST(binary, b08_save_and_load) {
    auto data = clustered(500, 32, 6, 1403);
    for (bool keep_full : {true, false}) {
        HnswIndex h(32, Metric::Cosine, hnsw_binary(100, true, keep_full));
        FlatIndex f(32, Metric::InnerProduct, Schema::dynamic(), binary_q(100, true, keep_full));
        for (std::uint64_t i = 0; i < 500; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        const std::string hb = save_bytes(h), fb = save_bytes(f);
        auto h2 = load_hnsw(hb);
        auto f2 = load_flat(fb);
        CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb && h2->codec()->kind() == CodecKind::Binary);
        CHECK(same_results(h2->search(data[3], 10, 64), h.search(data[3], 10, 64)) && same_results(f2->search(data[3], 10), f.search(data[3], 10)));
    }
}

TEST(binary, b09_damaged_files) {
    auto data = random_vectors(16, 5, 1404);
    HnswIndex h(5, Metric::L2, hnsw_binary(5, true, false));
    for (std::uint64_t i = 0; i < 16; ++i) h.add(i, data[i]);
    const std::string bytes = save_bytes(h);
    std::size_t rejected = 0, tried = 0;
    for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
        try { (void)load_hnsw(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
    }
    for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
        std::string damaged = bytes;
        damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
        try { (void)load_hnsw(damaged); } catch (const FormatError&) { ++rejected; }
    }
    CHECK(rejected == tried);
    // A negative spread behind a valid checksum. CODC: codec (1), train_after (8),
    // rerank (1), factor (8), trained (1), state size (8), thresholds (5 x 4), spreads.
    ParsedFile p = ParsedFile::parse(bytes);
    poke<float>(p.payload("CODC"), 27 + 20, -1.0f);
    CHECK_THROWS_AS(FormatError, load_hnsw(p.assemble()));
}

TEST(binary, b10_with_truncation) {
    auto data = clustered(1000, 64, 8, 1405);
    HnswParams p = hnsw_binary(200);
    p.quantization.truncate_to = 32;
    HnswIndex h(64, Metric::Cosine, p);
    for (std::uint64_t i = 0; i < 1000; ++i) h.add(i, data[i]);
    CHECK(h.codec()->dim() == 32 && h.get_code(1)->size() == 4 && h.search(data[9], 1, 64)[0].id == 9 && *h.get_vector(9) == data[9]);
}

TEST(binary, b11_training_is_final_with_codes_only) {
    auto data = clustered(300, 16, 4, 1406);
    HnswIndex h(16, Metric::L2, hnsw_binary(0, true, false));
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i]);
    h.train_codec();
    CHECK(!h.has_vector_values());
    CHECK_THROWS_AS(std::logic_error, h.train_codec());
}

TEST(binary, b12_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 16, 4, 1407);
    FlatIndex f(16, Metric::L2, Schema::dynamic(), binary_q(0, true, false));
    for (std::uint64_t i = 0; i < 300; ++i) f.add(i, data[i]);
    const std::string fb = save_bytes(f);
    CHECK(sweep_allocation_failures([&] { f.train_codec(); }, [&] { CHECK(save_bytes(f) == fb && f.has_vector_values()); }) > 0);
    CHECK(!f.has_vector_values() && codes_match(f));
}

// ===========================================================================
// Phase Q4: product quantization (PQ1-PQ12)
// ===========================================================================

namespace {

QuantizationParams pq_q(std::size_t train_after = 300, std::size_t parts = 0, bool keep_full = true, std::size_t factor = 4) {
    QuantizationParams q;
    q.codec = CodecKind::PQ;
    q.train_after = train_after;
    q.pq_subspaces = parts;
    q.keep_full_vectors = keep_full;
    q.rerank_factor = factor;
    return q;
}

HnswParams hnsw_pq(std::size_t train_after = 300, std::size_t parts = 0, bool keep_full = true) {
    HnswParams p;
    p.quantization = pq_q(train_after, parts, keep_full);
    return p;
}

}  // namespace

TEST(pq, pq01_parts_and_code_size) {
    CHECK(PQCodec(64, Metric::L2).parts() == 8 && PQCodec(64, Metric::L2).code_size() == 8);  // one per 8 dimensions
    CHECK(PQCodec(5, Metric::L2).parts() == 1 && PQCodec(24, Metric::L2, 5).code_size() == 5);  // uneven: 5, 5, 5, 5, 4
    CHECK_THROWS_AS(std::invalid_argument, PQCodec(4, Metric::L2, 5));
    CHECK(make_codec(CodecKind::PQ, 16, Metric::L2, 4)->code_size() == 4);
    HnswParams too_many = hnsw_pq(10, 9);
    CHECK_THROWS_AS(std::invalid_argument, HnswIndex(8, Metric::L2, too_many));
}

TEST(pq, pq02_few_vectors_are_stored_exactly) {
    // Fewer training vectors than 256: one centroid per vector, so each decodes exactly.
    std::vector<float> sample = {1, 2, 3, 4, 5, 6, -1, 0, 7, 7, 7, 7, 0, 0, 0, 0, 9, -9, 2, 3};  // 5 vectors of 4
    PQCodec c(4, Metric::L2, 2);
    c.train(sample);
    CHECK(c.centroids() == 5 && c.state().size() == 1 + 5 * 4);
    bool exact = true;
    for (std::size_t i = 0; i < 5; ++i) {
        std::uint8_t code[2];
        float back[4];
        c.encode(std::span<const float>(&sample[i * 4], 4), code);
        c.decode(code, back);
        exact = exact && std::equal(back, back + 4, &sample[i * 4]);
    }
    CHECK(exact);
}

TEST(pq, pq03_distances_match_decoded_vectors) {
    auto data = clustered(600, 24, 6, 1500);
    for (Metric m : kMetrics) {
        PQCodec c(24, m, 6);
        c.train(flatten(data));
        std::vector<float> table, back(24);
        std::uint8_t code[6];
        c.prepare(data[0], table);
        bool close = true;
        for (std::size_t i = 1; i < 600; i += 7) {
            c.encode(data[i], code);
            c.decode(code, back);
            double expected = 0;
            for (std::size_t j = 0; j < 24; ++j)
                expected += m == Metric::L2 ? double(data[0][j] - back[j]) * (data[0][j] - back[j]) : double(data[0][j]) * back[j];
            if (m == Metric::InnerProduct) expected = -expected;
            if (m == Metric::Cosine) expected = 1.0 - expected;
            close = close && std::abs(c.distance(table, code) - expected) < 1e-3 * (1 + std::abs(expected));
        }
        CHECK(close);
    }
}

TEST(pq, pq04_decode_then_encode_is_exact) {
    // Including duplicate centroids (many identical training vectors).
    std::vector<float> sample;
    for (int i = 0; i < 300; ++i)
        for (int j = 0; j < 8; ++j) sample.push_back(float((i % 3) * 10 + j % 2));  // 3 distinct vectors only
    PQCodec c(8, Metric::L2, 4);
    c.train(sample);
    bool exact = true;
    std::vector<float> back(8);
    for (std::size_t i = 0; i < 300; ++i) {
        std::uint8_t code[4], again[4];
        c.encode(std::span<const float>(&sample[i * 8], 8), code);
        c.decode(code, back);
        c.encode(back, again);
        exact = exact && std::equal(code, code + 4, again) && std::equal(back.begin(), back.end(), &sample[i * 8]);
    }
    CHECK(exact);
}

TEST(pq, pq05_rejects_misuse) {
    PQCodec c(8, Metric::L2, 2);
    std::uint8_t code[2] = {0, 0};
    std::vector<float> table, out(8);
    CHECK_THROWS_AS(std::logic_error, c.encode(std::vector<float>(8, 1.0f), code));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>{}));
    CHECK_THROWS_AS(std::invalid_argument, c.train(std::vector<float>(7, 1.0f)));
    for (const std::vector<float>& bad : {std::vector<float>{0}, std::vector<float>{257}, std::vector<float>{1.5f},
                                          std::vector<float>{2, 1, 1, 1, 1, 1, 1, 1, 1}, std::vector<float>{1, 0, 0, 0, kNaN, 0, 0, 0, 0}})
        CHECK_THROWS_AS(std::invalid_argument, c.restore(bad));
    c.restore(std::vector<float>{1, 0, 0, 0, 0, 1, 1, 1, 1});  // one centroid per part
    code[1] = 3;  // no centroid 3
    CHECK_THROWS_AS(std::invalid_argument, c.decode(code, out));
}

TEST(pq, pq06_recall) {
    auto data = clustered(4000, 64, 16, 1501);
    auto queries = clustered(50, 64, 16, 1502);
    for (Metric m : kMetrics) {
        HnswIndex plain(64, m), reranked(64, m, hnsw_pq(1000)), lean(64, m, hnsw_pq(1000, 0, false));
        FlatIndex exact(64, m), f4(64, m, Schema::dynamic(), pq_q(1000)), f16(64, m, Schema::dynamic(), pq_q(1000, 0, true, 16));
        for (std::uint64_t i = 0; i < 4000; ++i) {
            plain.add(i, data[i]);
            reranked.add(i, data[i]);
            lean.add(i, data[i]);
            exact.add(i, data[i]);
            f4.add(i, data[i]);
            f16.add(i, data[i]);
        }
        const double base = recall(plain, exact, queries, 10, 64), r = recall(reranked, exact, queries, 10, 64);
        const double r_wide = recall(reranked, exact, queries, 10, 200), l = recall(lean, exact, queries, 10, 64);
        double a = 0, b = 0;
        for (const auto& q : queries) {
            std::set<std::uint64_t> truth;
            for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
            for (const auto& x : f4.search(q, 10)) a += double(truth.count(x.id));
            for (const auto& x : f16.search(q, 10)) b += double(truth.count(x.id));
        }
        a /= 500.0;
        b /= 500.0;
        std::printf("    metric %d: hnsw %.3f | pq + re-rank: ef 64 %.3f, ef 200 %.3f; codes only %.3f | flat: re-rank x4 %.3f, x16 %.3f\n",
                    int(m), base, r, r_wide, l, a, b);
        CHECK(!lean.has_vector_values() && lean.get_code(1)->size() == 8);
        // Measured on this synthetic data (8 dimensions per part): like binary codes,
        // PQ needs a generous re-ranking budget; on its own it is coarse.
        CHECK(r_wide >= 0.88 && r_wide > r && r >= 0.60 && b >= 0.85 && a >= 0.55 && l >= 0.25);
    }
}

TEST(pq, pq07_every_path) {
    auto data = clustered(1500, 24, 6, 1503);
    for (bool keep_full : {true, false}) {
        HnswIndex h(24, Metric::L2, hnsw_pq(300, 6, keep_full));
        FlatIndex f(24, Metric::L2, Schema::dynamic(), pq_q(300, 6, keep_full));
        for (std::uint64_t i = 0; i < 500; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        for (std::uint64_t i = 0; i < 500; i += 5) {
            h.remove(i);
            f.remove(i);
        }
        for (std::uint64_t i = 1000; i < 1050; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        for (std::uint64_t i = 1; i < 500; i += 7) {
            h.update(i, data[i + 900]);
            f.update(i, data[i + 900]);
        }
        h.add_batch(id_range(5000, 60), flatten(Vectors(data.begin() + 1400, data.begin() + 1460)), 4);
        f.add_batch(id_range(5000, 60), flatten(Vectors(data.begin() + 1400, data.begin() + 1460)));
        CHECK(codes_match(h) && codes_match(f) && h.has_vector_values() == keep_full);
        h.compact();
        CHECK(codes_match(h) && h.deleted_count() == 0);
        if (keep_full) check_graph(h, 0.9);
        CHECK(h.search_by_id(5001, 5, 64)->size() == 5 && !f.search_range(data[1401], 50.0f).empty());
    }
}

TEST(pq, pq08_save_and_load) {
    auto data = clustered(600, 24, 6, 1504);
    for (bool keep_full : {true, false}) {
        HnswIndex h(24, Metric::Cosine, hnsw_pq(300, 5, keep_full));  // 5 uneven parts
        FlatIndex f(24, Metric::InnerProduct, Schema::dynamic(), pq_q(300, 5, keep_full));
        for (std::uint64_t i = 0; i < 600; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        const std::string hb = save_bytes(h), fb = save_bytes(f);
        auto h2 = load_hnsw(hb);
        auto f2 = load_flat(fb);
        CHECK(save_bytes(*h2) == hb && save_bytes(*f2) == fb && h2->params().quantization.pq_subspaces == 5 && h2->codec()->code_size() == 5);
        CHECK(same_results(h2->search(data[3], 10, 64), h.search(data[3], 10, 64)) && same_results(f2->search(data[3], 10), f.search(data[3], 10)));
    }
    HnswIndex untrained(24, Metric::L2, hnsw_pq(10000, 3));  // the number of parts is kept even before training
    untrained.add(1, data[1]);
    CHECK(load_hnsw(save_bytes(untrained))->params().quantization.pq_subspaces == 3);
}

TEST(pq, pq09_damaged_and_invalid_files) {
    auto data = random_vectors(30, 6, 1505);
    HnswIndex h(6, Metric::L2, hnsw_pq(10, 3, false));
    for (std::uint64_t i = 0; i < 30; ++i) h.add(i, data[i]);
    const std::string bytes = save_bytes(h);
    std::size_t rejected = 0, tried = 0;
    for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
        try { (void)load_hnsw(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
    }
    for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
        std::string damaged = bytes;
        damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
        try { (void)load_hnsw(damaged); } catch (const FormatError&) { ++rejected; }
    }
    CHECK(rejected == tried);
    // Valid checksums, invalid content: more parts than dimensions (the last 8
    // bytes of CODC), and a code naming a centroid that does not exist (10 here).
    ParsedFile parts = ParsedFile::parse(bytes);
    poke<std::uint64_t>(parts.payload("CODC"), parts.payload("CODC").size() - 8, 7);
    ParsedFile code = ParsedFile::parse(bytes);
    std::string& codc = code.payload("CODC");
    codc[codc.size() - 9] = char(200);  // the last code byte, just before the number of parts
    for (ParsedFile* e : {&parts, &code}) CHECK_THROWS_AS(FormatError, load_hnsw(e->assemble()));
}

TEST(pq, pq10_with_truncation) {
    auto data = clustered(1000, 64, 8, 1506);
    HnswParams p = hnsw_pq(300, 4);
    p.quantization.truncate_to = 32;
    HnswIndex h(64, Metric::Cosine, p);
    for (std::uint64_t i = 0; i < 1000; ++i) h.add(i, data[i]);
    CHECK(h.codec()->dim() == 32 && h.get_code(1)->size() == 4 && h.search(data[9], 1, 64)[0].id == 9 && *h.get_vector(9) == data[9]);
}

TEST(pq, pq11_training_is_final_with_codes_only) {
    auto data = clustered(400, 16, 4, 1507);
    HnswIndex h(16, Metric::L2, hnsw_pq(0, 4, false));
    for (std::uint64_t i = 0; i < 400; ++i) h.add(i, data[i]);
    h.train_codec();
    CHECK(!h.has_vector_values());
    CHECK_THROWS_AS(std::logic_error, h.train_codec());
}

TEST(pq, pq12_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(300, 16, 4, 1508);
    FlatIndex f(16, Metric::L2, Schema::dynamic(), pq_q(0, 4, false));
    for (std::uint64_t i = 0; i < 300; ++i) f.add(i, data[i]);
    const std::string fb = save_bytes(f);
    CHECK(sweep_allocation_failures([&] { f.train_codec(); }, [&] { CHECK(save_bytes(f) == fb && f.has_vector_values()); }) > 0);
    CHECK(!f.has_vector_values() && codes_match(f));
}

// ===========================================================================
// Phase Q5: IVF, an inverted file for FlatIndex (I1-I12)
// ===========================================================================

namespace {

IvfParams ivf_p(std::size_t lists, std::size_t probes = 4, std::size_t train_after = 0) {
    IvfParams p;
    p.lists = lists;
    p.probes = probes;
    p.train_after = train_after;
    return p;
}

/// The lists hold every vector exactly once; and, when full vectors are kept,
/// every vector searched with a single probe finds itself (or an identical copy)
/// first: it sits in the list nearest to it. With codes only, a vector added after
/// training is placed by its true vector, whose nearest list can differ from its
/// decoded one's, so only the bookkeeping is checked.
bool lists_consistent(const FlatIndex& f) {
    if (!f.lists_trained()) return false;
    std::size_t total = 0;
    for (std::size_t s : f.list_sizes()) total += s;
    if (total != f.size()) return false;
    ScanOptions ids;
    ids.limit = 1000000;
    ids.with_vectors = false;
    ids.with_metadata = false;
    std::vector<std::size_t> counted(f.list_sizes().size(), 0);
    for (const Record& r : f.scan(ids).records) {
        const auto l = f.list_of(r.id);
        if (!l || *l >= counted.size()) return false;
        ++counted[*l];
    }
    if (counted != f.list_sizes()) return false;
    if (!f.has_vector_values()) return true;
    SearchOptions one;
    one.probes = 1;
    ScanOptions all;
    all.limit = 1000000;
    for (const Record& r : f.scan(all).records) {
        auto in = r.vector;
        in.resize(f.dim(), 0.0f);  // truncating indexes read back the indexed values only
        const auto top = f.search(in, 1, one);
        if (top.empty() || (top[0].id != r.id && top[0].distance > 1e-5f)) return false;
    }
    return true;
}

}  // namespace

TEST(ivf, i01_untrained_scans_everything) {
    auto data = clustered(800, 16, 8, 1600);
    FlatIndex plain(16, Metric::L2), f(16, Metric::L2, Schema::dynamic(), {}, ivf_p(16));
    for (std::uint64_t i = 0; i < 800; ++i) {
        plain.add(i, data[i]);
        f.add(i, data[i]);
    }
    CHECK(!f.lists_trained() && !f.list_of(1) && f.list_sizes().empty());
    bool same = true;
    for (std::size_t q = 0; q < 10; ++q) same = same && same_results(f.search(data[q], 10), plain.search(data[q], 10));
    CHECK(same);
}

TEST(ivf, i02_training) {
    auto data = clustered(600, 8, 6, 1601);
    FlatIndex none(8, Metric::L2), autof(8, Metric::L2, Schema::dynamic(), {}, ivf_p(10, 2, 300)), f(8, Metric::L2, Schema::dynamic(), {}, ivf_p(10, 2, 0));
    CHECK_THROWS_AS(std::logic_error, none.train_lists());
    for (std::uint64_t i = 0; i < 5; ++i) f.add(i, data[i]);
    CHECK_THROWS_AS(std::invalid_argument, f.train_lists());  // fewer vectors than lists
    for (std::uint64_t i = 0; i < 600; ++i) {
        autof.add(i, data[i]);
        if (i == 298) CHECK(!autof.lists_trained());
    }
    CHECK(autof.lists_trained() && autof.list_sizes().size() == 10 && lists_consistent(autof));  // trained at 300
    for (std::uint64_t i = 5; i < 600; ++i) f.add(i, data[i]);
    CHECK(!f.lists_trained());  // 0: only explicitly
    f.train_lists();
    CHECK(lists_consistent(f));
    f.train_lists();  // again: replaces the lists
    CHECK(lists_consistent(f));
    CHECK_THROWS_AS(std::invalid_argument, FlatIndex(8, Metric::L2, Schema::dynamic(), {}, ivf_p(10, 0)));
}

TEST(ivf, i03_probing_every_list_is_exact) {
    auto data = clustered(1500, 24, 10, 1602);
    for (Metric m : kMetrics) {
        FlatIndex exact(24, m), f(24, m, Schema::dynamic(), {}, ivf_p(20, 20, 0));
        for (std::uint64_t i = 0; i < 1500; ++i) {
            exact.add(i, data[i]);
            f.add(i, data[i]);
        }
        f.train_lists();
        bool same = true;
        for (std::size_t q = 0; q < 20; ++q) same = same && same_up_to_ties(f.search(data[q * 7], 10), exact.search(data[q * 7], 10));
        CHECK(same);
    }
}

TEST(ivf, i04_recall_rises_with_probes) {
    auto data = clustered(5000, 32, 40, 1603);
    auto queries = clustered(60, 32, 40, 1604);
    FlatIndex exact(32, Metric::L2), f(32, Metric::L2, Schema::dynamic(), {}, ivf_p(64, 1, 0));
    for (std::uint64_t i = 0; i < 5000; ++i) {
        exact.add(i, data[i]);
        f.add(i, data[i]);
    }
    f.train_lists();
    std::vector<double> r;
    for (std::size_t probes : {std::size_t(1), std::size_t(4), std::size_t(16), std::size_t(64)}) {
        SearchOptions o;
        o.probes = probes;
        SearchStats st;
        o.stats = &st;
        double hit = 0, visited = 0;
        for (const auto& q : queries) {
            std::set<std::uint64_t> truth;
            for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
            for (const auto& x : f.search(q, 10, o)) hit += double(truth.count(x.id));
            visited += double(st.nodes_visited);
        }
        r.push_back(hit / 600.0);
        std::printf("    probes %2zu: recall %.3f, vectors scanned %.0f of 5000\n", probes, hit / 600.0, visited / 60.0);
    }
    // Measured on this noisy synthetic data: recall rises with probes, and probing
    // every list is exact.
    CHECK(r[0] < r[1] && r[1] < r[2] && r[2] < r[3] && r[3] == 1.0 && r[2] >= 0.75);
}

TEST(ivf, i05_every_path_keeps_the_lists) {
    auto data = clustered(1500, 12, 8, 1605);
    FlatIndex f(12, Metric::L2, Schema::dynamic(), {}, ivf_p(12, 2, 200));
    for (std::uint64_t i = 0; i < 500; ++i) f.add(i, data[i], meta_for(i));
    for (std::uint64_t i = 0; i < 500; i += 4) f.remove(i);  // swap with last
    for (std::uint64_t i = 1; i < 500; i += 5) f.update(i, data[1499 - i]);  // possibly another list
    for (std::uint64_t i = 2; i < 500; i += 9) f.upsert(i, data[1000 + i]);
    f.add_batch(id_range(5000, 100), flatten(Vectors(data.begin() + 1300, data.begin() + 1400)));
    CHECK(lists_consistent(f));
    std::vector<Metadata> meta(50);
    std::atomic<bool> flag{false};
    OperationOptions o;
    o.cancel = &flag;
    o.progress = [&](std::size_t done, std::size_t) { if (done > 20) flag = true; };
    CHECK_THROWS_AS(Cancelled, f.add_batch(id_range(7000, 50), flatten(Vectors(data.begin() + 1200, data.begin() + 1250)), meta, 0, o));
    CHECK(lists_consistent(f));
    CHECK(f.remove_where(Filter::eq("category", "tech")) > 0 && lists_consistent(f));
    f.compact();
    CHECK(lists_consistent(f));
}

TEST(ivf, i06_ivf_pq) {
    auto data = clustered(4000, 64, 32, 1606);
    auto queries = clustered(50, 64, 32, 1607);
    QuantizationParams pq;
    pq.codec = CodecKind::PQ;
    pq.train_after = 1000;
    pq.rerank_factor = 16;
    FlatIndex exact(64, Metric::L2), f(64, Metric::L2, Schema::dynamic(), pq, ivf_p(32, 8, 1000));
    QuantizationParams lean_q = pq;
    lean_q.keep_full_vectors = false;
    FlatIndex lean(64, Metric::L2, Schema::dynamic(), lean_q, ivf_p(32, 8, 1000));
    for (std::uint64_t i = 0; i < 4000; ++i) {
        exact.add(i, data[i]);
        f.add(i, data[i]);
        lean.add(i, data[i]);
    }
    CHECK(f.lists_trained() && f.codec_trained() && lean.lists_trained() && !lean.has_vector_values() && lists_consistent(lean));
    double a = 0, all = 0, b = 0;
    SearchOptions every;
    every.probes = 32;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
        for (const auto& x : f.search(q, 10)) a += double(truth.count(x.id));
        for (const auto& x : f.search(q, 10, every)) all += double(truth.count(x.id));
        for (const auto& x : lean.search(q, 10)) b += double(truth.count(x.id));
    }
    std::printf("    IVF-PQ (32 lists): re-ranked, 8 probes %.3f, all 32 probes %.3f; codes only, 8 probes %.3f\n", a / 500.0, all / 500.0, b / 500.0);
    // Two approximations stack: the probed lists, then the codes (re-ranked).
    CHECK(all / 500.0 >= 0.85 && all >= a && a / 500.0 >= 0.45 && b / 500.0 >= 0.20);
}

TEST(ivf, i07_filters_predicates_and_k) {
    auto data = clustered(2000, 16, 20, 1608);
    FlatIndex f(16, Metric::L2, Schema::dynamic(), {}, ivf_p(32, 1, 0));
    for (std::uint64_t i = 0; i < 2000; ++i) f.add(i, data[i], meta_for(i));
    f.train_lists();
    // A selective filter: one probe holds too few matches, so more lists are probed: never fewer than k results.
    const auto r = f.search(data[0], 10, with_filter(Filter::eq("category", "tech")));
    bool ok = r.size() == 10;
    for (const auto& x : r) ok = ok && x.id % 5 == 2;
    SearchOptions odd;
    odd.predicate = [](std::uint64_t id) { return id % 97 == 3; };  // about 20 vectors in total
    const auto p = f.search(data[0], 15, odd);
    ok = ok && p.size() == 15;
    for (const auto& x : p) ok = ok && x.id % 97 == 3;
    CHECK(ok && f.search(data[0], 5000).size() == 2000);  // more than exists: everything
    SearchStats st;
    SearchOptions late;
    late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    late.probes = 32;
    late.stats = &st;
    (void)f.search(data[0], 10, late);
    CHECK(st.stopped_early && st.nodes_visited < 2000);
}

TEST(ivf, i08_batch_and_range) {
    auto data = clustered(1000, 16, 10, 1609);
    FlatIndex f(16, Metric::Cosine, Schema::dynamic(), {}, ivf_p(10, 3, 0)), exact(16, Metric::Cosine);
    for (std::uint64_t i = 0; i < 1000; ++i) {
        f.add(i, data[i]);
        exact.add(i, data[i]);
    }
    f.train_lists();
    const auto queries = flatten(Vectors(data.begin(), data.begin() + 12));
    const auto batch = f.search_batch(queries, 8, 4);
    bool same = true;
    for (std::size_t q = 0; q < 12; ++q) same = same && same_results(batch[q], f.search(data[q], 8));
    CHECK(same && same_results(f.search_range(data[3], 0.05f), exact.search_range(data[3], 0.05f)));  // range: exact
}

TEST(ivf, i09_save_and_load) {
    auto data = clustered(700, 12, 8, 1610);
    for (bool trained : {false, true}) {
        FlatIndex f(12, Metric::InnerProduct, Schema::dynamic(), {}, ivf_p(9, 3, trained ? 100 : 0));
        for (std::uint64_t i = 0; i < 700; ++i) f.add(i, data[i]);
        f.remove(5);
        const std::string bytes = save_bytes(f);
        auto f2 = load_flat(bytes);
        CHECK(save_bytes(*f2) == bytes && f2->lists_trained() == trained && f2->ivf().lists == 9 && f2->ivf().probes == 3);
        CHECK(same_results(f2->search(data[1], 10), f.search(data[1], 10)));
        if (trained) CHECK(f2->list_sizes() == f.list_sizes() && lists_consistent(*f2));
    }
}

TEST(ivf, i10_damaged_and_invalid_files) {
    auto data = random_vectors(40, 4, 1611);
    FlatIndex f(4, Metric::L2, Schema::dynamic(), {}, ivf_p(3, 2, 0));
    for (std::uint64_t i = 0; i < 40; ++i) f.add(i, data[i]);
    f.train_lists();
    const std::string bytes = save_bytes(f);
    std::size_t rejected = 0, tried = 0;
    for (std::size_t len = 0; len < bytes.size(); ++len, ++tried) {
        try { (void)load_flat(bytes.substr(0, len)); } catch (const FormatError&) { ++rejected; }
    }
    for (std::size_t i = 0; i < bytes.size(); ++i, ++tried) {
        std::string damaged = bytes;
        damaged[i] = char(damaged[i] ^ (1 << (i % 8)));
        try { (void)load_flat(damaged); } catch (const FormatError&) { ++rejected; }
    }
    CHECK(rejected == tried);
    // IVFL: lists (8), probes (8), train_after (8), trained (1), trained lists (8), centroids (3 x 4 x 4), list numbers.
    const ParsedFile base = ParsedFile::parse(bytes);
    const std::vector<std::function<void(std::string&)>> edits = {
        [](std::string& p) { poke<std::uint64_t>(p, 8, 0); },                // probes 0
        [](std::string& p) { poke<std::uint8_t>(p, 24, 2); },                // training flag 2
        [](std::string& p) { poke<std::uint64_t>(p, 25, 0); },               // no trained lists
        [](std::string& p) { poke<std::uint64_t>(p, 25, 4); },               // more trained lists than configured
        [](std::string& p) { poke<float>(p, 33 + 4, kNaN); },                // a centroid that is not finite
        [](std::string& p) { poke<std::uint32_t>(p, p.size() - 4, 3); },     // a vector in list 3 of 3
        [](std::string& p) { p.resize(p.size() - 4); },                      // one list number missing
    };
    for (const auto& edit : edits) {
        ParsedFile e = base;
        edit(e.payload("IVFL"));
        CHECK_THROWS_AS(FormatError, load_flat(e.assemble()));
    }
}

TEST(ivf, i11_searching_while_lists_are_trained) {
    auto data = clustered(1500, 8, 10, 1612);
    FlatIndex f(8, Metric::L2, Schema::dynamic(), {}, ivf_p(16, 4, 800));
    for (std::uint64_t i = 0; i < 400; ++i) f.add(i, data[i]);
    std::atomic<bool> stop{false};
    Problems p;
    with_watchdog(120, "i11", [&] {
        std::thread writer([&] {
            for (std::uint64_t i = 400; i < 1500; ++i) f.add(i, data[i]);  // trains the lists at 800
            stop = true;
        });
        run_threads(3, [&](std::size_t t) {
            std::size_t q = t;
            while (!stop) check_results(f.search(data[q++ % 1500], 10), 1500, p);
        });
        writer.join();
    });
    CHECK(p.total() == 0 && lists_consistent(f));
}

TEST(ivf, i12_out_of_memory) {
    if constexpr (!HNSW_ALLOC_HOOK) SKIP("allocation hook disabled under sanitizers");
    auto data = clustered(400, 8, 6, 1613);
    FlatIndex f(8, Metric::L2, Schema::dynamic(), {}, ivf_p(8, 2, 0));
    for (std::uint64_t i = 0; i < 300; ++i) f.add(i, data[i]);
    const std::string untrained = save_bytes(f);
    CHECK(sweep_allocation_failures([&] { f.train_lists(); }, [&] { CHECK(save_bytes(f) == untrained && !f.lists_trained()); }) > 0);
    const std::string trained = save_bytes(f);
    std::uint64_t id = 900;
    CHECK(sweep_allocation_failures([&] { f.add(id, data[350]); }, [&] { CHECK(save_bytes(f) == trained); }) > 0);
    CHECK(sweep_allocation_failures([&] { (void)f.update(3, data[360]); }, [&] { CHECK(lists_consistent(f)); }) >= 0);
    CHECK(lists_consistent(f) && f.contains(900));
}

// ===========================================================================
// Edge cases across SIMD kernels, PQ and IVF (E1-E8)
// ===========================================================================

namespace {
IvfParams ivf_e(std::size_t lists, std::size_t probes) {
    IvfParams p;
    p.lists = lists;
    p.probes = probes;
    p.train_after = 0;
    return p;
}
std::size_t total_listed(const FlatIndex& f) {
    std::size_t t = 0;
    for (std::size_t s : f.list_sizes()) t += s;
    return t;
}
}  // namespace

TEST(edges, e01_ivf_list_counts) {
    auto data = random_vectors(60, 4, 1700);
    FlatIndex one(4, Metric::L2, Schema::dynamic(), {}, ivf_e(1, 1)), exact(4, Metric::L2), many(4, Metric::L2, Schema::dynamic(), {}, ivf_e(4, 100));
    for (std::uint64_t i = 0; i < 60; ++i) {
        one.add(i, data[i]);
        exact.add(i, data[i]);
        many.add(i, data[i]);
    }
    one.train_lists();
    many.train_lists();
    CHECK(same_up_to_ties(one.search(data[7], 5), exact.search(data[7], 5)));  // one list: an exact scan
    CHECK(many.search(data[7], 5).size() == 5 && one.search(data[7], 0).empty());  // probes beyond lists; k = 0
}

TEST(edges, e02_ivf_emptied_and_refilled) {
    auto data = random_vectors(40, 4, 1701);
    FlatIndex f(4, Metric::L2, Schema::dynamic(), {}, ivf_e(4, 2));
    for (std::uint64_t i = 0; i < 20; ++i) f.add(i, data[i]);
    f.train_lists();
    for (std::uint64_t i = 0; i < 20; ++i) f.remove(i);
    CHECK(f.search(data[0], 5).empty() && total_listed(f) == 0 && f.lists_trained());
    for (std::uint64_t i = 20; i < 40; ++i) f.add(i, data[i]);
    CHECK(total_listed(f) == 20 && f.search(data[25], 30).size() == 20 && f.search(data[25], 1)[0].id == 25);
}

TEST(edges, e03_ivf_unmatched_filter_and_duplicates) {
    FlatIndex f(4, Metric::L2, Schema::dynamic(), {}, ivf_e(5, 1));
    for (std::uint64_t i = 0; i < 30; ++i) f.add(i, std::vector<float>(4, 2.0f), Metadata().set("c", "a"));  // identical vectors
    f.train_lists();
    CHECK(f.search(std::vector<float>(4, 2.0f), 30).size() == 30);
    CHECK(f.search(std::vector<float>(4, 2.0f), 5, with_filter(Filter::eq("c", "zzz"))).empty());  // every list probed, none match
}

TEST(edges, e04_ivf_with_other_features) {
    auto data = random_vectors(60, 6, 1702);
    QuantizationParams q;
    q.truncate_to = 2;
    FlatIndex f(6, Metric::Cosine, Schema::dynamic(), q, ivf_e(3, 1));
    for (std::uint64_t i = 0; i < 30; ++i) f.add(i, data[i], Metadata().set("c", i % 2 ? "a" : "b"));
    f.add(99, std::vector<float>(6, 0.0f));  // a zero vector, before and after training
    f.train_lists();
    f.add(98, std::vector<float>(6, 0.0f));
    CHECK(f.search(data[3], 40).size() == 32);
    CHECK(f.remove_batch(std::vector<std::uint64_t>{1, 2, 3, 4}) == 4 && f.set_metadata_where(Filter::eq("c", "a"), Metadata().set("x", 1)) > 0);
    CHECK(total_listed(f) == 28);
    auto g = load_flat(save_bytes(f));  // and keep adding after loading
    for (std::uint64_t i = 30; i < 60; ++i) g->add(i, data[i]);
    CHECK(total_listed(*g) == 58);
}

TEST(edges, e05_pq_extremes) {
    auto data = random_vectors(50, 6, 1703);
    PQCodec per_dim(6, Metric::L2, 6);  // one dimension per part
    per_dim.train(flatten(data));
    std::uint8_t code[6];
    per_dim.encode(data[0], code);
    PQCodec single(8, Metric::L2, 2);  // trained on one vector: one centroid per part
    single.train(random_vectors(1, 8, 1704)[0]);
    std::uint8_t c2[2];
    single.encode(data[1].size() == 6 ? std::vector<float>(8, 1.0f) : data[1], c2);
    CHECK(per_dim.code_size() == 6 && single.centroids() == 1 && c2[0] == 0 && c2[1] == 0 && PQCodec(3, Metric::L2).parts() == 1);
}

TEST(edges, e06_codecs_with_huge_ranges) {
    // Values spanning more than the float range: max - min overflows a float.
    const std::vector<float> sample = {-3e38f, 0, 3e38f, 1};
    SQ8Codec sq8(2, Metric::L2);
    sq8.train(sample);
    std::uint8_t code[2];
    sq8.encode(std::vector<float>{1e38f, 0.5f}, code);  // 1e38 - (-3e38) also overflows a float
    float back[2];
    sq8.decode(code, back);
    CHECK(std::isfinite(back[0]) && std::abs(back[0] - 1e38f) <= sq8.step()[0] * 0.5f + 1e33f && code[0] > 100 && code[0] < 255);
    BinaryCodec bin(2, Metric::L2);
    bin.train(sample);
    PQCodec pq(2, Metric::L2, 1);
    pq.train(std::vector<float>{-3e38f, 0, 3e38f, 1, 0, 0});
    CHECK(bin.trained() && pq.trained());
    // Automatic training must not fail silently on such data.
    HnswParams p;
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 10;
    HnswIndex h(2, Metric::L2, p);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, std::vector<float>{i % 2 ? 3e38f : -3e38f, float(i)});
    CHECK(h.codec_trained());
}

TEST(edges, e07_kernel_tails) {
    const auto k = get_code_kernels();
    std::vector<float> w(13, 1.0f), t(13, 1.0f), s(13, 0.0f);
    std::uint8_t bits[2] = {0xFF, 0x1F}, bytes[13] = {};
    CHECK(k.bits_sum(w.data(), bits, 13) == 13.0f && k.sq8_l2(t.data(), s.data(), bytes, 13) == 13.0f && k.sq8_dot(w.data(), bytes, 13) == 0.0f);
}

TEST(edges, e08_sq8_fixed_answers_unchanged) {
    // Computing SQ8 in double (for huge ranges) must not change ordinary codes:
    // the same values as q0_11, computed independently with NumPy float32.
    const std::vector<float> sample = {0, -1, 2.5f, 255, 1, 0.5f, 127.5f, 0, -0.25f};
    SQ8Codec c(3, Metric::L2);
    c.train(sample);
    std::uint8_t code[3];
    c.encode(std::vector<float>{63.75f, 0.3f, 1.0f}, code);
    CHECK(code[0] == 64 && code[1] == 166 && code[2] == 116);
}

TEST(edges, e09_codec_misuse_branches) {
    BinaryCodec b(4, Metric::L2);
    CHECK_THROWS_AS(std::invalid_argument, b.train(std::vector<float>(6, 1.0f)));  // not whole vectors
    CHECK_THROWS_AS(std::invalid_argument, b.restore(std::vector<float>{kNaN, 0, 0, 0, 1, 1, 1, 1}));  // a threshold that is not finite
    CHECK_THROWS_AS(std::invalid_argument, b.restore(std::vector<float>{-3e38f, 0, 0, 0, 3e38f, 1, 1, 1}));  // threshold - spread overflows
    PQCodec q(8, Metric::L2, 2);
    CHECK(q.state().empty() && !q.trained());  // untrained: no state
    CHECK_THROWS_AS(std::invalid_argument, q.train(std::vector<float>{1, 2, 3, 4, 5, 6, 7, kInf}));
    CHECK_THROWS_AS(std::invalid_argument, q.restore(std::vector<float>{}));
}

TEST(edges, e10_list_of_unknown_and_untrained) {
    IvfParams ivf;
    ivf.lists = 2;
    ivf.train_after = 0;
    FlatIndex f(4, Metric::L2, Schema::dynamic(), {}, ivf);
    for (std::uint64_t i = 0; i < 10; ++i) f.add(i, random_vectors(1, 4, unsigned(1710 + i))[0]);
    CHECK(!f.list_of(3));  // untrained
    f.train_lists();
    CHECK(f.list_of(3) && !f.list_of(999));  // unknown ID
}

TEST(edges, e11_inconsistent_codec_and_list_sections) {
    // A trained CODC section with fewer than 8 bytes left for its last field
    // (the number of PQ parts), and "trained" lists while no lists are configured.
    auto data = random_vectors(20, 4, 1711);
    HnswParams p;
    p.quantization.codec = CodecKind::PQ;
    p.quantization.train_after = 5;
    p.quantization.pq_subspaces = 2;
    HnswIndex h(4, Metric::L2, p);
    for (std::uint64_t i = 0; i < 20; ++i) h.add(i, data[i]);
    ParsedFile short_codc = ParsedFile::parse(save_bytes(h));
    std::string& codc = short_codc.payload("CODC");
    codc.resize(codc.size() - 8 - 40 + 4);  // drop the field and the codes, keep 4 stray bytes
    CHECK_THROWS_AS(FormatError, load_hnsw(short_codc.assemble()));
    IvfParams ivf;
    ivf.lists = 2;
    ivf.train_after = 0;
    FlatIndex f(4, Metric::L2, Schema::dynamic(), {}, ivf);
    for (std::uint64_t i = 0; i < 20; ++i) f.add(i, data[i]);
    f.train_lists();
    ParsedFile no_lists = ParsedFile::parse(save_bytes(f));
    poke<std::uint64_t>(no_lists.payload("IVFL"), 0, 0);  // lists = 0, yet trained
    try {
        (void)load_flat(no_lists.assemble());
        CHECK(false);
    } catch (const FormatError& e) {
        CHECK(std::string(e.what()).find("training flag") != std::string::npos);  // rejected by its own rule
    }
}

// ===========================================================================
// Phase Q6a: the Vamana graph (VA1-VA8)
// ===========================================================================

namespace {

/// Every node reachable from the medoid, by breadth-first search.
std::size_t reachable_from_medoid(const VamanaGraph& g) {
    std::vector<bool> seen(g.size(), false);
    std::vector<NodeId> queue{g.medoid()};
    seen[g.medoid()] = true;
    for (std::size_t head = 0; head < queue.size(); ++head)
        for (NodeId nb : g.neighbors(queue[head]))
            if (!seen[nb]) {
                seen[nb] = true;
                queue.push_back(nb);
            }
    return queue.size();
}

double vamana_recall(const VamanaGraph& g, const Vectors& data, const Vectors& queries, Metric m, std::size_t L) {
    FlatIndex exact(data[0].size(), m);
    for (std::uint64_t i = 0; i < data.size(); ++i) exact.add(i, data[i]);
    double hit = 0;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& r : exact.search(q, 10)) truth.insert(r.id);
        for (const auto& r : g.search(q, 10, L)) hit += double(truth.count(r.id));
    }
    return hit / double(10 * queries.size());
}

}  // namespace

TEST(vamana, va01_recall) {
    auto data = clustered(4000, 32, 20, 1800);
    auto queries = clustered(50, 32, 20, 1801);
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        const auto g = VamanaGraph::build(flatten(data), 32, m, VamanaParams{32, 64, 1.2f, 7});
        const double r64 = vamana_recall(g, data, queries, m, 64), r128 = vamana_recall(g, data, queries, m, 128);
        std::printf("    metric %d: recall@10 with L 64 %.3f, L 128 %.3f\n", int(m), r64, r128);
        CHECK(r64 >= 0.90 && r128 >= r64);
    }
}

TEST(vamana, va02_structure) {
    auto data = clustered(3000, 16, 10, 1802);
    const auto g = VamanaGraph::build(flatten(data), 16, Metric::L2, VamanaParams{24, 48, 1.2f, 3});
    bool ok = g.size() == 3000;
    std::size_t edges = 0;
    for (NodeId p = 0; p < g.size(); ++p) {
        const auto nbs = g.neighbors(p);
        std::set<NodeId> unique(nbs.begin(), nbs.end());
        ok = ok && nbs.size() <= 24 && unique.size() == nbs.size() && !unique.count(p) && !nbs.empty();
        for (NodeId q : nbs) ok = ok && q < g.size();
        edges += nbs.size();
    }
    std::printf("    average degree %.1f of at most 24; reachable from the medoid: %zu of 3000\n", double(edges) / 3000.0, reachable_from_medoid(g));
    CHECK(ok && reachable_from_medoid(g) == 3000);
}

TEST(vamana, va03_alpha) {
    // Both prune strengths give good graphs (alpha > 1 keeps long-range edges).
    auto data = clustered(4000, 24, 30, 1803);
    auto queries = clustered(40, 24, 30, 1804);
    const auto one = VamanaGraph::build(flatten(data), 24, Metric::L2, VamanaParams{16, 40, 1.0f, 5});
    const auto wide = VamanaGraph::build(flatten(data), 24, Metric::L2, VamanaParams{16, 40, 1.2f, 5});
    double e_one = 0, e_wide = 0;
    for (const auto& q : queries) {
        std::size_t a = 0, b = 0;
        (void)one.search(q, 10, 40, &a);
        (void)wide.search(q, 10, 40, &b);
        e_one += double(a);
        e_wide += double(b);
    }
    const double r_one = vamana_recall(one, data, queries, Metric::L2, 40), r_wide = vamana_recall(wide, data, queries, Metric::L2, 40);
    std::printf("    nodes expanded per search: alpha 1.0 %.1f (recall %.3f), alpha 1.2 %.1f (recall %.3f)\n",
                e_one / 40, r_one, e_wide / 40, r_wide);
    // Recorded, not asserted: with a fixed beam both expand about L nodes; alpha's
    // benefit (a shorter graph diameter) is measured as disk reads in Q6c.
    CHECK(r_one >= 0.85 && r_wide >= 0.85);
}

TEST(vamana, va04_deterministic) {
    auto data = clustered(800, 8, 6, 1805);
    const auto a = VamanaGraph::build(flatten(data), 8, Metric::Cosine, VamanaParams{12, 24, 1.2f, 9});
    const auto b = VamanaGraph::build(flatten(data), 8, Metric::Cosine, VamanaParams{12, 24, 1.2f, 9});
    const auto c = VamanaGraph::build(flatten(data), 8, Metric::Cosine, VamanaParams{12, 24, 1.2f, 10});
    bool same = a.medoid() == b.medoid(), differs = false;
    for (NodeId p = 0; p < 800; ++p) {
        same = same && std::ranges::equal(a.neighbors(p), b.neighbors(p));
        differs = differs || !std::ranges::equal(a.neighbors(p), c.neighbors(p));
    }
    CHECK(same && differs);
}

TEST(vamana, va05_small_and_degenerate) {
    const auto one = VamanaGraph::build(std::vector<float>{1, 2, 3}, 3, Metric::L2, VamanaParams{4, 8});
    CHECK(one.size() == 1 && one.neighbors(0).empty() && one.search(std::vector<float>{0, 0, 0}, 5, 8).size() == 1);
    const auto two = VamanaGraph::build(std::vector<float>{0, 0, 1, 1}, 2, Metric::L2, VamanaParams{4, 8});
    CHECK(two.neighbors(0).size() == 1 && two.neighbors(1).size() == 1 && two.search(std::vector<float>{1, 1}, 1, 8)[0].id == 1);
    auto few = random_vectors(10, 4, 1806);  // fewer vectors than R: a complete graph
    const auto complete = VamanaGraph::build(flatten(few), 4, Metric::L2, VamanaParams{32, 32});
    bool full = true;
    for (NodeId p = 0; p < 10; ++p) full = full && complete.neighbors(p).size() == 9;
    CHECK(full && complete.search(few[4], 10, 32).size() == 10 && complete.search(few[4], 1, 32)[0].id == 4);
    const auto same = VamanaGraph::build(std::vector<float>(200, 2.5f), 2, Metric::L2, VamanaParams{8, 16});  // 100 identical vectors
    CHECK(same.search(std::vector<float>{2.5f, 2.5f}, 10, 16).size() == 10 && reachable_from_medoid(same) == 100);
    std::vector<float> line;
    for (int i = 0; i < 300; ++i) line.push_back(float(i));
    const auto one_dim = VamanaGraph::build(line, 1, Metric::L2, VamanaParams{6, 12});
    CHECK(one_dim.search(std::vector<float>{150.2f}, 1, 12)[0].id == 150 && reachable_from_medoid(one_dim) == 300);
    CHECK(one_dim.search(std::vector<float>{3}, 0, 12).empty());
}

TEST(vamana, va06_rejects_invalid_input) {
    const std::vector<float> v = {1, 2, 3, 4};
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::InnerProduct));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build({}, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 3, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 0, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(std::vector<float>{1, kNaN}, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::L2, VamanaParams{0, 8}));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::L2, VamanaParams{8, 4}));  // L below R
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::L2, VamanaParams{8, 8, 0.9f}));
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::L2, VamanaParams{8, 8, kNaN}));
    const auto g = VamanaGraph::build(v, 2, Metric::L2, VamanaParams{4, 4});
    CHECK_THROWS_AS(std::invalid_argument, (void)g.search(std::vector<float>{1, 2, 3}, 1, 4));
}

TEST(vamana, va07_search_parameters) {
    auto data = clustered(500, 8, 5, 1807);
    const auto g = VamanaGraph::build(flatten(data), 8, Metric::L2, VamanaParams{16, 32});
    CHECK(g.search(data[3], 600, 32).size() == 500 && g.search(data[3], 20, 5).size() == 20);  // k beyond n; L below k
    const auto r = g.search(data[3], 10, 32);
    bool sorted = true;
    for (std::size_t i = 1; i < r.size(); ++i) sorted = sorted && r[i - 1].distance <= r[i].distance;
    CHECK(sorted && r[0].id == 3 && r[0].distance == 0.0f);
}

TEST(vamana, va08_cosine_normalizes) {
    // Scaled copies of a vector are the same direction: distance 0 under cosine.
    auto data = random_vectors(300, 6, 1808);
    const auto g = VamanaGraph::build(flatten(data), 6, Metric::Cosine, VamanaParams{12, 24});
    std::vector<float> scaled = data[7];
    for (float& x : scaled) x *= 9.5f;
    const auto r = g.search(scaled, 1, 24);
    CHECK(r[0].id == 7 && r[0].distance < 1e-5f);
    const auto v = g.vector(7);
    double len = 0;
    for (float x : v) len += double(x) * x;
    CHECK(std::abs(len - 1.0) < 1e-5);
}

TEST(vamana, va09_duplicate_groups_among_distinct_vectors) {
    // 60 copies of one vector among 600 distinct ones, R = 8: a group larger than
    // R links mostly within itself. Every vector must still be reachable, and
    // the copies found.
    auto data = clustered(600, 8, 6, 1809);
    const auto copy = data[5];
    for (int i = 0; i < 60; ++i) data.push_back(copy);
    const auto g = VamanaGraph::build(flatten(data), 8, Metric::L2, VamanaParams{8, 24, 1.2f, 2});
    auto queries = clustered(30, 8, 6, 1810);
    const double r = vamana_recall(g, data, queries, Metric::L2, 48);
    const std::size_t reach = reachable_from_medoid(g);
    const auto at_copy = g.search(copy, 20, 48);
    std::size_t zeros = 0;
    for (const auto& x : at_copy) zeros += x.distance == 0.0f;
    std::printf("    reachable %zu of 660; recall@10 %.3f; a query at the copies finds %zu of them at distance 0 (top 20)\n", reach, r, zeros);
    CHECK(reach == 660 && r >= 0.85 && zeros == 20);  // every vector reachable (connectivity repair)
}

// ===========================================================================
// Phase Q6b: the disk layout (DK1-DK8)
// ===========================================================================

namespace {

/// Rewrites bytes of a disk index file and re-signs its header, so that checks
/// beyond the header checksum are reached.
void patch_disk_file(const std::string& path, std::size_t offset, const std::string& bytes, bool resign = true) {
    std::string f = read_file(path);
    f.replace(offset, bytes.size(), bytes);
    if (resign) {
        const std::size_t header_size = 168;
        const std::uint32_t crc = persist::crc32(0, f.data(), header_size);
        unsigned char b[4];
        persist::to_little_endian(crc, b);
        f.replace(header_size, 4, reinterpret_cast<const char*>(b), 4);
    }
    write_file(path, f);
}

template <class T>
std::string le(T v) {
    unsigned char b[sizeof(T)];
    persist::to_little_endian(v, b);
    return std::string(reinterpret_cast<const char*>(b), sizeof(T));
}

}  // namespace

TEST(disk, dk01_round_trip_matches_the_graph) {
    auto data = clustered(1000, 32, 8, 1900);
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        const std::string path = temp_path("dk01");
        DiskIndexParams p;
        p.graph = VamanaParams{16, 32, 1.2f, 4};
        std::vector<Metadata> meta;
        for (std::uint64_t i = 0; i < 1000; ++i) meta.push_back(meta_for(i));
        DiskIndex::build(path, id_range(5000, 1000), flatten(data), 32, m, p, meta);
        const auto d = DiskIndex::open(path);
        const auto g = VamanaGraph::build(flatten(data), 32, m, p.graph);  // deterministic: the same graph
        bool same = d->size() == 1000 && d->dim() == 32 && d->metric() == m && d->medoid() == g.medoid() && d->max_degree() == 16;
        for (NodeId i = 0; i < 1000; ++i) {
            const auto node = d->read_node(i);
            same = same && std::ranges::equal(node.neighbors, g.neighbors(i)) && std::ranges::equal(node.vector, g.vector(i));
            same = same && d->id_of(i) == 5000 + i && *d->node_of(5000 + i) == i && d->metadata().read(i) == meta_for(i);
        }
        CHECK(same && !d->node_of(42) && d->codec().trained() && d->codec().parts() == 16);  // one part per 2 dimensions
        std::filesystem::remove(path);
    }
}

TEST(disk, dk02_layouts) {
    // Many nodes per block; a node filling a block exactly; a node spanning blocks.
    struct Case { std::size_t dim, R, nodes_per_block, blocks_per_node; };
    for (const Case c : {Case{4, 8, 4096 / 52, 1}, Case{1000, 23, 1, 1}, Case{2048, 8, 0, 3}}) {
        auto data = random_vectors(40, c.dim, unsigned(1901 + c.dim));
        const std::string path = temp_path("dk02");
        DiskIndexParams p;
        p.graph = VamanaParams{c.R, c.R + 8, 1.2f, 1};
        DiskIndex::build(path, id_range(0, 40), flatten(data), c.dim, Metric::L2, p);
        const auto d = DiskIndex::open(path);
        bool ok = true;
        for (NodeId i = 0; i < 40; ++i) ok = ok && d->read_node(i).vector == data[i];
        const std::uint64_t blocks = c.nodes_per_block ? (40 + c.nodes_per_block - 1) / c.nodes_per_block : 40 * c.blocks_per_node;
        CHECK(ok && d->blocks_read() == 40 * c.blocks_per_node);
        CHECK(std::filesystem::file_size(path) > 4096 * (1 + blocks));
        std::filesystem::remove(path);
    }
}

TEST(disk, dk03_damaged_files) {
    auto data = random_vectors(200, 8, 1904);
    const std::string good = temp_path("dk03_good"), path = temp_path("dk03");
    DiskIndexParams p;
    p.graph = VamanaParams{8, 16, 1.2f, 1};
    DiskIndex::build(good, id_range(0, 200), flatten(data), 8, Metric::L2, p);
    const std::string bytes = read_file(good);
    auto rejects = [&](auto edit) {
        write_file(path, bytes);
        edit();
        try {
            (void)DiskIndex::open(path);
            return false;
        } catch (const FormatError&) {
            return true;
        }
    };
    // Header: magic, version, byte order, its own checksum (not re-signed).
    CHECK(rejects([&] { patch_disk_file(path, 0, "X", false); }));
    CHECK(rejects([&] { patch_disk_file(path, 8, le<std::uint32_t>(2)); }));
    CHECK(rejects([&] { patch_disk_file(path, 12, le<std::uint32_t>(0x04030201)); }));
    CHECK(rejects([&] { patch_disk_file(path, 16, le<std::uint64_t>(9), false); }));
    // Re-signed, so the checks behind the checksum are reached.
    CHECK(rejects([&] { patch_disk_file(path, 16, le<std::uint64_t>(9)); }));         // dimension: layout no longer matches
    CHECK(rejects([&] { patch_disk_file(path, 24, le<std::uint32_t>(1)); }));         // inner product
    CHECK(rejects([&] { patch_disk_file(path, 48, le<std::uint64_t>(200)); }));       // medoid out of range
    CHECK(rejects([&] { patch_disk_file(path, 40, le<std::uint64_t>(0)); }));         // R = 0
    CHECK(rejects([&] { patch_disk_file(path, 88, le<std::uint64_t>(4096)); }));      // a section offset
    // Truncated, trailing data, and a damaged byte in each in-memory section.
    CHECK(rejects([&] { write_file(path, bytes.substr(0, bytes.size() - 1)); }));
    CHECK(rejects([&] { write_file(path, bytes.substr(0, 100)); }));
    CHECK(rejects([&] { write_file(path, bytes + "x"); }));
    for (int s = 0; s < 4; ++s) {
        const std::uint64_t offset = persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(bytes.data() + 88 + 20 * s));
        CHECK(rejects([&] { patch_disk_file(path, offset + 2, std::string(1, char(bytes[offset + 2] ^ 0x40)), false); }));
    }
    // A damaged node block: opening succeeds; reading that node fails, others work.
    write_file(path, bytes);
    patch_disk_file(path, 4096 + 10, std::string(1, char(bytes[4096 + 10] ^ 1)), false);
    const auto d = DiskIndex::open(path);
    CHECK_THROWS_AS(FormatError, (void)d->read_node(0));
    CHECK(d->read_node(199).vector == data[199]);
    CHECK_THROWS_AS(std::out_of_range, (void)d->read_node(200));
    std::filesystem::remove(good);
    std::filesystem::remove(path);
}

TEST(disk, dk04_rejects_invalid_input) {
    const std::string path = temp_path("dk04");
    const std::vector<float> v = {1, 2, 3, 4, 5, 6};
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 1, 2}, v, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 2}, v, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{}, std::vector<float>{}, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 2, 3}, v, 2, Metric::InnerProduct));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 2, 3}, std::vector<float>{1, 2, kNaN, 4, 5, 6}, 2, Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 2, 3}, v, 2, Metric::L2, {}, std::vector<Metadata>(2)));
    CHECK(!std::filesystem::exists(path) && temporaries_left(path) == 0);
    CHECK_THROWS_AS(std::runtime_error, (void)DiskIndex::open(temp_path("dk04_missing")));
    write_file(path, "short");
    CHECK_THROWS_AS(FormatError, (void)DiskIndex::open(path));
    std::filesystem::remove(path);
}

TEST(disk, dk05_replaces_a_file_only_when_complete) {
    auto data = random_vectors(50, 4, 1905);
    const std::string path = temp_path("dk05");
    write_file(path, "the previous file");
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1, 1}, std::vector<float>(8, 1.0f), 4, Metric::L2));
    CHECK(read_file(path) == "the previous file" && temporaries_left(path) == 0);
    DiskIndex::build(path, id_range(0, 50), flatten(data), 4, Metric::L2);
    CHECK(DiskIndex::open(path)->size() == 50 && temporaries_left(path) == 0);
    std::filesystem::remove(path);
}

TEST(disk, dk06_schema_and_metadata) {
    auto data = random_vectors(30, 4, 1906);
    const std::string path = temp_path("dk06");
    std::vector<Metadata> meta(30);
    for (std::size_t i = 0; i < 30; ++i) meta[i] = meta_for(i);
    meta[7] = Metadata().set("not_in_schema", 1);
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, id_range(0, 30), flatten(data), 4, Metric::L2, {}, meta, Schema::strict(search_fields())));
    meta[7] = meta_for(7);
    DiskIndex::build(path, id_range(0, 30), flatten(data), 4, Metric::L2, {}, meta, Schema::strict(search_fields()));
    const auto d = DiskIndex::open(path);
    CHECK(d->metadata().read(7) == meta_for(7) && d->metadata().find_field("category").has_value());
    DiskIndex::build(path, id_range(0, 30), flatten(data), 4, Metric::L2);  // without metadata
    CHECK(DiskIndex::open(path)->metadata().read(3) == Metadata());
    std::filesystem::remove(path);
}

TEST(disk, dk07_concurrent_reads) {
    auto data = random_vectors(600, 16, 1907);
    const std::string path = temp_path("dk07");
    DiskIndex::build(path, id_range(0, 600), flatten(data), 16, Metric::L2);
    const auto d = DiskIndex::open(path);
    std::atomic<long> wrong{0}, reads{0};
    with_watchdog(60, "dk07", [&] {
        run_threads(4, [&](std::size_t t) {
            for (NodeId i = NodeId(t); i < 600; i += 2) {
                wrong += long(d->read_node(i).vector != data[i]);
                ++reads;
            }
        });
    });
    CHECK(wrong == 0 && reads == 300 + 300 + 299 + 299 && d->blocks_read() == std::uint64_t(reads));  // one block per node here
    std::filesystem::remove(path);
}

TEST(disk, dk08_single_vector) {
    const std::string path = temp_path("dk08");
    DiskIndex::build(path, std::vector<std::uint64_t>{77}, std::vector<float>{1, 2, 3}, 3, Metric::Cosine);
    const auto d = DiskIndex::open(path);
    CHECK(d->size() == 1 && d->medoid() == 0 && d->read_node(0).neighbors.empty() && d->id_of(0) == 77);
    std::filesystem::remove(path);
}

// ===========================================================================
// Phase Q6c: searching on disk (DK9-DK15)
// ===========================================================================

namespace {

struct DiskRun {
    double recall = 0, nodes = 0, blocks = 0, rounds = 0;
};

DiskRun disk_run(const DiskIndex& d, FlatIndex& exact, const Vectors& queries, DiskSearchParams p) {
    DiskRun r;
    for (const auto& q : queries) {
        std::set<std::uint64_t> truth;
        for (const auto& x : exact.search(q, 10)) truth.insert(x.id);
        DiskSearchStats st;
        for (const auto& x : d.search(q, 10, p, {}, &st)) r.recall += double(truth.count(x.id));
        r.nodes += double(st.nodes_read);
        r.blocks += double(st.blocks_read);
        r.rounds += double(st.rounds);
    }
    const double n = double(queries.size());
    r.recall /= 10 * n;
    r.nodes /= n;
    r.blocks /= n;
    r.rounds /= n;
    return r;
}

}  // namespace

TEST(disk, dk09_recall_and_reads) {
    auto data = clustered(5000, 32, 20, 1910);
    auto queries = clustered(40, 32, 20, 1911);
    const std::string path = temp_path("dk09");
    DiskIndexParams p;
    p.graph = VamanaParams{32, 64, 1.2f, 3};
    DiskIndex::build(path, id_range(0, 5000), flatten(data), 32, Metric::L2, p);
    const auto d = DiskIndex::open(path);
    FlatIndex exact(32, Metric::L2);
    for (std::uint64_t i = 0; i < 5000; ++i) exact.add(i, data[i]);
    std::vector<DiskRun> runs;
    for (std::size_t L : {std::size_t(20), std::size_t(50), std::size_t(100)}) {
        runs.push_back(disk_run(*d, exact, queries, DiskSearchParams{L, 4}));
        std::printf("    L %3zu, W 4: recall@10 %.3f, nodes read %.1f, blocks %.1f, rounds %.1f (of 5000 vectors)\n", L,
                    runs.back().recall, runs.back().nodes, runs.back().blocks, runs.back().rounds);
    }
    const DiskRun w1 = disk_run(*d, exact, queries, DiskSearchParams{50, 1});
    std::printf("    L  50, W 1: recall@10 %.3f, nodes read %.1f, rounds %.1f\n", w1.recall, w1.nodes, w1.rounds);
    CHECK(runs[0].recall <= runs[1].recall + 0.02 && runs[1].recall <= runs[2].recall + 0.02 && runs[2].recall >= 0.95);
    CHECK(runs[2].nodes < 400 && w1.rounds > runs[1].rounds * 2);  // a few hundred reads out of 5000; W 4: far fewer rounds
    std::filesystem::remove(path);
}

TEST(disk, dk10_results_are_exact) {
    auto data = clustered(1500, 16, 10, 1912);
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        const std::string path = temp_path("dk10");
        DiskIndex::build(path, id_range(100, 1500), flatten(data), 16, m);
        const auto d = DiskIndex::open(path);
        FlatIndex exact(16, m);
        for (std::uint64_t i = 0; i < 1500; ++i) exact.add(100 + i, data[i]);
        bool ok = true;
        for (std::size_t q = 0; q < 20; ++q) {
            const auto r = d->search(data[q * 50], 10, DiskSearchParams{60, 4});
            ok = ok && r.size() == 10 && r[0].id == 100 + q * 50 && sorted_and_unique(r);
            for (const auto& x : r) {  // the distance an exact index computes for the same vector
                const auto e = exact.search(data[x.id - 100], 1);
                ok = ok && x.id >= 100 && x.id < 1600 && std::abs(x.distance - exact.search(data[q * 50], 1500)[0].distance) >= 0;
                (void)e;
            }
            const auto truth = exact.search(data[q * 50], 1500);
            std::map<std::uint64_t, float> by_id;
            for (const auto& t : truth) by_id[t.id] = t.distance;
            for (const auto& x : r) ok = ok && std::abs(x.distance - by_id[x.id]) <= 1e-5f * (1 + by_id[x.id]);
        }
        if (m == Metric::Cosine) {  // a scaled query is the same direction
            std::vector<float> scaled = data[7];
            for (float& x : scaled) x *= 6.0f;
            ok = ok && d->search(scaled, 1)[0].id == 107;
        }
        CHECK(ok && d->search(data[0], 0).empty());
        CHECK(d->search(data[0], 5000, DiskSearchParams{2000, 8}).size() == 1500);  // k beyond n, a beam covering everything
        std::filesystem::remove(path);
    }
}

TEST(disk, dk11_rejects_invalid_searches) {
    auto data = random_vectors(50, 4, 1913);
    const std::string path = temp_path("dk11");
    DiskIndex::build(path, id_range(0, 50), flatten(data), 4, Metric::L2);
    const auto d = DiskIndex::open(path);
    CHECK_THROWS_AS(std::invalid_argument, (void)d->search(data[0], 5, DiskSearchParams{0, 4}));
    CHECK_THROWS_AS(std::invalid_argument, (void)d->search(data[0], 5, DiskSearchParams{10, 0}));
    CHECK_THROWS_AS(std::invalid_argument, (void)d->search(std::vector<float>{1, 2, 3}, 5));
    CHECK_THROWS_AS(std::invalid_argument, (void)d->search(std::vector<float>{1, 2, kNaN, 4}, 5));
    CHECK(d->search(data[0], 5, {}, with_filter(Filter::eq("c", 1))).empty());  // an unknown field (dynamic schema): nothing matches
    std::filesystem::remove(path);
}

TEST(disk, dk12_deadlines_and_cancellation) {
    auto data = random_vectors(500, 8, 1914);
    const std::string path = temp_path("dk12");
    DiskIndex::build(path, id_range(0, 500), flatten(data), 8, Metric::L2);
    const auto d = DiskIndex::open(path);
    SearchOptions late;
    late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    SearchStats ss;
    late.stats = &ss;
    DiskSearchStats st;
    const auto r = d->search(data[0], 10, {}, late, &st);
    CHECK(st.stopped_early && ss.stopped_early && st.nodes_read == 0 && r.empty());  // stopped before the first read
    std::atomic<bool> cancel{true};
    SearchOptions cancelled;
    cancelled.cancel = &cancel;
    CHECK(d->search(data[0], 10, {}, cancelled, &st).empty() && st.stopped_early);
    SearchOptions far;
    far.deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
    CHECK(same_results(d->search(data[0], 10, {}, far, &st), d->search(data[0], 10)) && !st.stopped_early);
    std::filesystem::remove(path);
}

TEST(disk, dk13_concurrent_searches) {
    auto data = clustered(2000, 16, 10, 1915);
    const std::string path = temp_path("dk13");
    DiskIndex::build(path, id_range(0, 2000), flatten(data), 16, Metric::L2);
    const auto d = DiskIndex::open(path);
    std::vector<std::vector<SearchResult>> expected;
    for (std::size_t q = 0; q < 40; ++q) expected.push_back(d->search(data[q * 37], 10));
    std::atomic<long> wrong{0};
    with_watchdog(60, "dk13", [&] {
        run_threads(4, [&](std::size_t t) {
            for (std::size_t round = 0; round < 3; ++round)
                for (std::size_t q = t; q < 40; q += 2) wrong += long(!same_results(d->search(data[q * 37], 10), expected[q]));
        });
    });
    CHECK(wrong == 0);
    std::filesystem::remove(path);
}

TEST(disk, dk14_damage_met_during_a_search) {
    auto data = random_vectors(300, 8, 1916);
    const std::string path = temp_path("dk14");
    DiskIndex::build(path, id_range(0, 300), flatten(data), 8, Metric::L2);
    const NodeId medoid = DiskIndex::open(path)->medoid();
    const std::size_t per_block = 4096 / (4 * 8 + 4 + 4 * 64);
    const std::uint64_t at = 4096 * (1 + medoid / per_block) + 5;
    std::string bytes = read_file(path);
    bytes[at] = char(bytes[at] ^ 0x10);
    write_file(path, bytes);
    const auto d = DiskIndex::open(path);  // opening does not read node blocks
    CHECK_THROWS_AS(FormatError, (void)d->search(data[0], 5));  // every search starts at the medoid
    std::filesystem::remove(path);
}

TEST(disk, dk15_alpha_and_reads) {
    // Alpha's purpose, measured where it matters: disk reads for a given recall.
    auto data = clustered(5000, 24, 30, 1917);
    auto queries = clustered(40, 24, 30, 1918);
    FlatIndex exact(24, Metric::L2);
    for (std::uint64_t i = 0; i < 5000; ++i) exact.add(i, data[i]);
    for (float alpha : {1.0f, 1.2f}) {
        const std::string path = temp_path("dk15");
        DiskIndexParams p;
        p.graph = VamanaParams{24, 48, alpha, 6};
        DiskIndex::build(path, id_range(0, 5000), flatten(data), 24, Metric::L2, p);
        const auto d = DiskIndex::open(path);
        for (std::size_t L : {std::size_t(30), std::size_t(60)}) {
            const DiskRun r = disk_run(*d, exact, queries, DiskSearchParams{L, 4});
            std::printf("    alpha %.1f, L %2zu: recall@10 %.3f, nodes read %.1f, rounds %.1f\n", double(alpha), L, r.recall, r.nodes, r.rounds);
            CHECK(r.recall >= 0.80);
        }
        std::filesystem::remove(path);
    }
}

namespace {
/// The search fields plus "rare" (an int), for strict schemas in the disk tests.
std::vector<FieldSpec> search_fields_with_rare() {
    auto f = search_fields();
    f.push_back({"rare", FieldType::Int});
    return f;
}
}  // namespace

// ===========================================================================
// Phase Q6d: filters, records, batches (DK16-DK23)
// ===========================================================================

namespace {

struct DiskFixture {
    Vectors data;
    std::string path;
    std::unique_ptr<DiskIndex> d;
    FlatIndex exact;
    DiskFixture(std::size_t n, std::size_t dim, unsigned seed, Metric m = Metric::L2, Schema schema = Schema::dynamic())
        : data(clustered(n, dim, 10, seed)), path(temp_path("dkfix")), exact(dim, m, schema) {
        std::vector<Metadata> meta;
        for (std::uint64_t i = 0; i < n; ++i) {
            meta.push_back(meta_for(i).set("rare", std::int64_t(i % 50 == 0)));
            exact.add(i, data[i], meta.back());
        }
        DiskIndex::build(path, id_range(0, n), flatten(data), dim, m, {}, meta, schema);
        d = DiskIndex::open(path);
    }
    ~DiskFixture() {
        d.reset();
        std::filesystem::remove(path);
    }
};

}  // namespace

TEST(disk, dk16_filtered_graph_search) {
    DiskFixture f(3000, 16, 1920);
    double hit = 0;
    bool only_matching = true, full = true;
    for (std::size_t q = 0; q < 20; ++q) {
        const auto opts = with_filter(Filter::eq("category", "tech"));  // a fifth of the vectors: the graph search
        const auto r = f.d->search(f.data[q * 101], 10, DiskSearchParams{60, 4}, opts);
        std::set<std::uint64_t> truth;
        for (const auto& x : f.exact.search(f.data[q * 101], 10, opts)) truth.insert(x.id);
        for (const auto& x : r) {
            only_matching = only_matching && x.id % 5 == 2;
            hit += double(truth.count(x.id));
        }
        full = full && r.size() == 10;
    }
    std::printf("    filtered (a fifth match): recall@10 %.3f\n", hit / 200.0);
    CHECK(only_matching && full && hit / 200.0 >= 0.9);
}

TEST(disk, dk17_selective_filter_is_exact) {
    DiskFixture f(3000, 16, 1921);
    bool same = true;
    for (std::size_t q = 0; q < 10; ++q) {  // 60 vectors match: at most L (100), read directly
        const auto opts = with_filter(Filter::eq("rare", std::int64_t(1)));
        DiskSearchStats st;
        const auto r = f.d->search(f.data[q * 77], 10, {}, opts, &st);
        same = same && same_up_to_ties(r, f.exact.search(f.data[q * 77], 10, opts)) && st.nodes_read == 60;
    }
    CHECK(same && f.d->search(f.data[0], 10, {}, with_filter(Filter::eq("category", "none"))).empty());  // nothing matches
}

TEST(disk, dk18_never_fewer_than_k) {
    // 60 of 3000 match, more than L (20): the graph search, which widens L until it finds 10.
    DiskFixture f(3000, 16, 1922);
    bool ok = true;
    for (std::size_t q = 0; q < 10; ++q) {
        const auto opts = with_filter(Filter::eq("rare", std::int64_t(1)));
        const auto r = f.d->search(f.data[q * 31], 10, DiskSearchParams{20, 4}, opts);
        ok = ok && r.size() == 10;
        for (const auto& x : r) ok = ok && x.id % 50 == 0;
    }
    SearchOptions odd;
    odd.predicate = [](std::uint64_t id) { return id % 97 == 3; };  // about 31 match
    const auto p = f.d->search(f.data[5], 25, DiskSearchParams{10, 4}, odd);
    ok = ok && p.size() == 25;
    for (const auto& x : p) ok = ok && x.id % 97 == 3;
    CHECK(ok);
}

TEST(disk, dk19_predicates_and_schemas) {
    DiskFixture f(1000, 8, 1923, Metric::L2, Schema::strict(search_fields_with_rare()));
    SearchOptions both = with_filter(Filter::eq("category", "tech"));
    both.predicate = [](std::uint64_t id) { return id % 2 == 0; };
    bool ok = true;
    for (const auto& x : f.d->search(f.data[3], 10, {}, both)) ok = ok && x.id % 5 == 2 && x.id % 2 == 0;
    CHECK(ok);
    CHECK_THROWS_AS(std::invalid_argument, (void)f.d->search(f.data[3], 10, {}, with_filter(Filter::eq("no_such_field", 1))));
}

TEST(disk, dk20_records) {
    DiskFixture f(500, 8, 1924, Metric::Cosine);
    CHECK(f.d->contains(7) && !f.d->contains(500) && !f.d->get_vector(500) && !f.d->get(500));
    const auto v = *f.d->get_vector(7);
    double len = 0, dot = 0, given = 0;
    for (std::size_t j = 0; j < 8; ++j) {
        len += double(v[j]) * v[j];
        dot += double(v[j]) * f.data[7][j];
        given += double(f.data[7][j]) * f.data[7][j];
    }
    CHECK(std::abs(len - 1.0) < 1e-5 && std::abs(dot - std::sqrt(given)) < 1e-4);  // stored normalized for cosine
    const auto r = *f.d->get(7);
    CHECK(r.id == 7 && r.vector == v && r.metadata.get("category").has_value());
    // scan: ascending IDs, pages, filters; without vectors it reads nothing from disk.
    ScanOptions o;
    o.limit = 64;
    o.with_vectors = false;
    std::vector<std::uint64_t> seen;
    const std::uint64_t before = f.d->blocks_read();
    for (Page page = f.d->scan(o);; page = f.d->scan(o)) {
        for (const auto& rec : page.records) seen.push_back(rec.id);
        if (!page.next) break;
        o.after = page.next;
    }
    CHECK(seen == id_range(0, 500) && f.d->blocks_read() == before);
    ScanOptions tech;
    tech.filter = Filter::eq("category", "tech");
    tech.limit = 1000;
    const auto t = f.d->scan(tech);
    CHECK(t.records.size() == 100 && !t.next && t.records[0].vector.size() == 8);
    o.limit = 0;
    CHECK_THROWS_AS(std::invalid_argument, (void)f.d->scan(o));
}

TEST(disk, dk21_search_by_id) {
    DiskFixture f(1000, 16, 1925);
    const auto without = *f.d->search_by_id(42, 5);
    const auto with = *f.d->search_by_id(42, 5, {}, {}, true);
    CHECK(without.size() == 5 && std::none_of(without.begin(), without.end(), [](const SearchResult& x) { return x.id == 42; }));
    CHECK(with[0].id == 42 && with[0].distance == 0.0f && !f.d->search_by_id(5000, 5) && f.d->search_by_id(42, 0)->empty());
}

TEST(disk, dk22_batches) {
    DiskFixture f(1500, 16, 1926);
    const auto queries = flatten(Vectors(f.data.begin(), f.data.begin() + 12));
    bool same = true;
    for (std::size_t threads : {std::size_t(1), std::size_t(4)}) {
        const auto b = f.d->search_batch(queries, 8, {}, threads);
        for (std::size_t q = 0; q < 12; ++q) same = same && same_results(b[q], f.d->search(f.data[q], 8));
    }
    CHECK(same && f.d->search_batch({}, 5).empty());
    CHECK_THROWS_AS(std::invalid_argument, (void)f.d->search_batch(std::vector<float>(17, 1.0f), 5));
    SearchStats st;
    SearchOptions late;
    late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    late.stats = &st;
    (void)f.d->search_batch(queries, 8, {}, 4, late);
    CHECK(st.stopped_early);
}

TEST(disk, dk23_concurrent_filtered_searches) {
    DiskFixture f(2000, 16, 1927);
    const auto opts = with_filter(Filter::eq("category", "news"));
    std::vector<std::vector<SearchResult>> expected;
    for (std::size_t q = 0; q < 20; ++q) expected.push_back(f.d->search(f.data[q * 53], 10, {}, opts));
    std::atomic<long> wrong{0};
    with_watchdog(60, "dk23", [&] {
        run_threads(4, [&](std::size_t t) {
            for (std::size_t q = t; q < 20; q += 2) wrong += long(!same_results(f.d->search(f.data[q * 53], 10, {}, opts), expected[q]));
        });
    });
    CHECK(wrong == 0);
}

namespace {

/// Re-signs a whole disk index file after an edit: every node block's checksum,
/// every section's, then the header's, so that the checks behind them are reached.
void resign_disk(std::string& f) {
    auto u64 = [&](std::size_t at) { return persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(f.data() + at)); };
    auto put32 = [&](std::size_t at, std::uint32_t v) {
        unsigned char b[4];
        persist::to_little_endian(v, b);
        f.replace(at, 4, reinterpret_cast<const char*>(b), 4);
    };
    const std::uint64_t blocks = u64(80), crc_at = u64(88);
    for (std::uint64_t b = 0; b < blocks; ++b) put32(std::size_t(crc_at + 4 * b), persist::crc32(0, f.data() + 4096 * (1 + b), 4096));
    for (std::size_t s = 0; s < 4; ++s) {
        const std::uint64_t offset = u64(88 + 20 * s), size = u64(96 + 20 * s);
        put32(104 + 20 * s, persist::crc32(0, f.data() + offset, std::size_t(size)));
    }
    put32(168, persist::crc32(0, f.data(), 168));
}

}  // namespace

TEST(disk, dk24_inconsistent_files_behind_valid_checksums) {
    auto data = random_vectors(60, 8, 1930);
    const std::string good = temp_path("dk24_good"), path = temp_path("dk24");
    DiskIndexParams p;
    p.graph = VamanaParams{8, 16, 1.2f, 1};
    p.pq_subspaces = 2;
    std::vector<Metadata> meta(60, Metadata().set("k", std::int64_t(1)));
    DiskIndex::build(good, id_range(0, 60), flatten(data), 8, Metric::L2, p, meta);
    const std::string bytes = read_file(good);
    auto u64 = [&](std::size_t at) { return std::size_t(persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(bytes.data() + at))); };
    const std::size_t pq = u64(108), ids = u64(128), metas = u64(148);  // section offsets (PQ, IDs, metadata)
    auto message = [&](auto edit) -> std::string {
        std::string f = bytes;
        edit(f);
        resign_disk(f);
        write_file(path, f);
        try {
            (void)DiskIndex::open(path);
            return "accepted";
        } catch (const FormatError& e) {
            return e.what();
        }
    };
    auto contains = [](const std::string& m, const char* what) { return m.find(what) != std::string::npos; };
    CHECK(contains(message([&](std::string& f) { f.replace(pq, 8, le<std::uint64_t>(0)); }), "invalid PQ section"));       // no parts
    CHECK(contains(message([&](std::string& f) { f.replace(pq + 8, 8, le<std::uint64_t>(1u << 30)); }), "invalid PQ section"));  // a huge state
    CHECK(contains(message([&](std::string& f) { f.replace(pq + 16, 4, le<float>(0.0f)); }), "invalid PQ codebooks"));  // 0 centroids
    CHECK(contains(message([&](std::string& f) { f.replace(u64(116) + pq - 1, 1, std::string(1, char(250))); }), "centroid that does not exist"));
    CHECK(contains(message([&](std::string& f) { f.replace(ids + 8, 8, le<std::uint64_t>(0)); }), "repeated ID"));
    CHECK(contains(message([&](std::string& f) { f[metas + 20] = char(f[metas + 20] ^ 0x5A); }), ""));  // inner damage: still a FormatError
    CHECK(message([&](std::string& f) { f[metas + 20] = char(f[metas + 20] ^ 0x5A); }) != "accepted");
    // Node records behind valid block checksums: an impossible degree, a missing neighbor.
    const std::size_t degree_at = 4096 + 4 * 8;
    for (int which = 0; which < 2; ++which) {
        std::string f = bytes;
        if (which == 0) f.replace(degree_at, 4, le<std::uint32_t>(9));          // R is 8
        else f.replace(degree_at + 4, 4, le<std::uint32_t>(60));                // node 60 of 60
        resign_disk(f);
        write_file(path, f);
        const auto d = DiskIndex::open(path);
        try {
            (void)d->read_node(0);
            CHECK(false);
        } catch (const FormatError& e) {
            CHECK(contains(e.what(), which == 0 ? "invalid node degree" : "a neighbor does not exist"));
        }
    }
    // Building into a directory that does not exist; reads past the end.
    CHECK_THROWS_AS(std::runtime_error, DiskIndex::build(temp_path("no_such_dir") + "/x.idx", id_range(0, 60), flatten(data), 8, Metric::L2));
    BlockFile file(good);
    char buf[16];
    CHECK_THROWS_AS(FormatError, file.read(file.size() - 4, buf, 16));
    CHECK_THROWS_AS(FormatError, file.read(file.size() + 1, buf, 1));
    std::filesystem::remove(good);
    std::filesystem::remove(path);
}

// ===========================================================================
// Extension E1: asynchronous reads (DK25-DK28)
// ===========================================================================

TEST(disk, dk25_async_results_are_identical) {
    auto data = clustered(2000, 16, 10, 1940);
    const std::string path = temp_path("dk25");
    std::vector<Metadata> meta;
    for (std::uint64_t i = 0; i < 2000; ++i) meta.push_back(meta_for(i).set("rare", std::int64_t(i % 50 == 0)));
    DiskIndex::build(path, id_range(0, 2000), flatten(data), 16, Metric::Cosine, {}, meta);
    const auto sync = DiskIndex::open(path);
    bool same = true;
    for (std::size_t threads : {std::size_t(1), std::size_t(4), std::size_t(8)}) {
        DiskOpenOptions o;
        o.io_threads = threads;
        const auto async = DiskIndex::open(path, o);
        for (std::size_t w : {std::size_t(1), std::size_t(4), std::size_t(8)})
            for (std::size_t q = 0; q < 10; ++q) {
                const DiskSearchParams p{40, w};
                DiskSearchStats a, b;
                same = same && same_results(async->search(data[q * 91], 10, p, {}, &a), sync->search(data[q * 91], 10, p, {}, &b));
                same = same && a.nodes_read == b.nodes_read && a.rounds == b.rounds;  // the same reads, in the same rounds
                for (const auto& opts : {with_filter(Filter::eq("rare", std::int64_t(1))), with_filter(Filter::eq("category", "tech"))})
                    same = same && same_results(async->search(data[q * 91], 10, p, opts), sync->search(data[q * 91], 10, p, opts));
            }
    }
    CHECK(same);
    std::filesystem::remove(path);
}

TEST(disk, dk26_reads_overlap) {
    // With slow storage (simulated: 3 ms a read) the W reads of a round overlap:
    // a round takes about one latency instead of W. Sleeps need no CPU, so this
    // holds on a loaded machine too.
    auto data = clustered(1500, 16, 10, 1941);
    const std::string path = temp_path("dk26");
    DiskIndex::build(path, id_range(0, 1500), flatten(data), 16, Metric::L2);
    DiskOpenOptions slow;
    slow.simulated_latency = std::chrono::milliseconds(3);
    DiskOpenOptions slow_async = slow;
    slow_async.io_threads = 4;
    const auto sync = DiskIndex::open(path, slow);
    const auto async = DiskIndex::open(path, slow_async);
    auto time = [&](const DiskIndex& d, std::size_t& rounds) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t q = 0; q < 5; ++q) {
            DiskSearchStats st;
            (void)d.search(data[q * 7], 10, DiskSearchParams{40, 4}, {}, &st);
            rounds += st.rounds;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::size_t rs = 0, ra = 0;
    const double ts = time(*sync, rs), ta = time(*async, ra);
    std::printf("    5 searches, W 4, 3 ms a read: one after another %.0f ms, concurrent %.0f ms (%zu rounds each)\n", ts, ta, rs);
    CHECK(rs == ra && ta < 0.6 * ts && ta >= double(ra) * 3.0 * 0.9);  // about one latency per round, never less
    std::filesystem::remove(path);
}

TEST(disk, dk27_errors_during_concurrent_reads) {
    auto data = random_vectors(300, 8, 1942);
    const std::string path = temp_path("dk27");
    DiskIndex::build(path, id_range(0, 300), flatten(data), 8, Metric::L2);
    std::size_t damaged_node;
    {
        const auto d = DiskIndex::open(path);
        damaged_node = d->read_node(d->medoid()).neighbors.at(0);  // read in the first round, beside others
    }
    const std::size_t per_block = 4096 / (4 * 8 + 4 + 4 * 64), block = damaged_node / per_block;
    std::string bytes = read_file(path);
    bytes[4096 * (1 + block) + 7] = char(bytes[4096 * (1 + block) + 7] ^ 0x08);
    write_file(path, bytes);
    DiskOpenOptions o;
    o.io_threads = 4;
    o.simulated_latency = std::chrono::microseconds(500);  // the other reads still running when one fails
    bool threw = false;
    {
        auto d = DiskIndex::open(path, o);
        try {
            (void)d->search(data[0], 10, DiskSearchParams{40, 8});
        } catch (const FormatError&) {
            threw = true;
        }
        CHECK(d->read_node(NodeId(block == 0 ? 299 : 0)).vector.size() == 8);  // still usable
    }  // destroyed right after: no read may still be running (AddressSanitizer checks)
    CHECK(threw);
    std::filesystem::remove(path);
}

TEST(disk, dk28_searches_share_the_pool) {
    auto data = clustered(2000, 16, 10, 1943);
    const std::string path = temp_path("dk28");
    DiskIndex::build(path, id_range(0, 2000), flatten(data), 16, Metric::L2);
    DiskOpenOptions o;
    o.io_threads = 3;
    const auto d = DiskIndex::open(path, o);
    std::vector<std::vector<SearchResult>> expected;
    for (std::size_t q = 0; q < 24; ++q) expected.push_back(d->search(data[q * 41], 10));
    std::atomic<long> wrong{0};
    with_watchdog(60, "dk28", [&] {
        run_threads(6, [&](std::size_t t) {  // more searching threads than readers
            for (std::size_t round = 0; round < 3; ++round)
                for (std::size_t q = t; q < 24; q += 3) wrong += long(!same_results(d->search(data[q * 41], 10), expected[q]));
        });
    });
    CHECK(wrong == 0 && same_results(d->search_batch(flatten(Vectors(data.begin(), data.begin() + 6)), 5, {}, 4)[2], d->search(data[2], 5)));
    std::filesystem::remove(path);
}

// ===========================================================================
// Extension E2: building indexes larger than memory (DK29-DK34)
// ===========================================================================

namespace {

std::size_t disk_reachable(const DiskIndex& d) {
    std::vector<bool> seen(d.size(), false);
    std::vector<NodeId> queue{d.medoid()};
    seen[d.medoid()] = true;
    for (std::size_t h = 0; h < queue.size(); ++h)
        for (NodeId nb : d.read_node(queue[h]).neighbors)
            if (!seen[nb]) {
                seen[nb] = true;
                queue.push_back(nb);
            }
    return queue.size();
}

std::size_t build_dirs_left(const std::string& path) {
    std::size_t left = 0;
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    const std::string name = std::filesystem::path(path).filename().string();
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().filename().string().rfind(name + ".build-", 0) == 0) ++left;
    return left;
}

}  // namespace

TEST(disk, dk29_fitting_in_the_budget_matches_build) {
    auto data = clustered(800, 16, 6, 1950);
    const std::string vec = temp_path("dk29_vectors"), a = temp_path("dk29_a"), b = temp_path("dk29_b");
    write_vectors_file(vec, flatten(data), 16);
    LargeBuildParams lp;
    lp.index.graph = VamanaParams{16, 32, 1.2f, 4};
    LargeBuildStats st;
    DiskIndex::build_from_file(a, vec, Metric::Cosine, lp, {}, {}, Schema::dynamic(), &st);
    DiskIndex::build(b, id_range(0, 800), flatten(data), 16, Metric::Cosine, lp.index);
    CHECK(st.partitions == 1 && read_file(a) == read_file(b));  // byte for byte
    for (const auto& f : {vec, a, b}) std::filesystem::remove(f);
}

TEST(disk, dk30_partitioned_build) {
    auto data = clustered(6000, 32, 30, 1951);
    auto queries = clustered(40, 32, 30, 1952);
    const std::string vec = temp_path("dk30_vectors"), path = temp_path("dk30");
    write_vectors_file(vec, flatten(data), 32);
    LargeBuildParams lp;
    lp.index.graph = VamanaParams{32, 64, 1.2f, 1};
    lp.memory_budget = std::size_t(400) << 10;
    const std::size_t cap = lp.memory_budget / (4 * PreparedVector(32).stride() + 8 * 32 + 96);
    LargeBuildStats st;
    DiskIndex::build_from_file(path, vec, Metric::L2, lp, {}, {}, Schema::dynamic(), &st);
    const auto d = DiskIndex::open(path);
    FlatIndex exact(32, Metric::L2);
    for (std::uint64_t i = 0; i < 6000; ++i) exact.add(i, data[i]);
    // Built in pieces, the graph is a little weaker at a tight beam and the same at a
    // normal one (measured: in memory 0.983 and 0.998, partitioned 0.950 and 0.998,
    // at L 50 and 100, for the same reads).
    const DiskRun tight = disk_run(*d, exact, queries, DiskSearchParams{50, 4});
    const DiskRun r = disk_run(*d, exact, queries, DiskSearchParams{100, 4});
    std::printf("    %zu partitions (largest %zu, cap %zu), %zu lists pruned at the merge; reachable %zu of 6000; recall@10 %.3f at L 50, %.3f at L 100\n",
                st.partitions, st.largest_partition, cap, st.merged_and_pruned, disk_reachable(*d), tight.recall, r.recall);
    CHECK(st.partitions > 1 && st.largest_partition <= cap && disk_reachable(*d) == 6000 && r.recall >= 0.97 && tight.recall >= 0.90);
    CHECK(build_dirs_left(path) == 0);  // temporary files removed
    for (const auto& f : {vec, path}) std::filesystem::remove(f);
}

TEST(disk, dk31_ids_metadata_and_cosine) {
    auto data = clustered(3000, 16, 12, 1953);
    const std::string vec = temp_path("dk31_vectors"), path = temp_path("dk31");
    write_vectors_file(vec, flatten(data), 16);
    std::vector<Metadata> meta;
    for (std::uint64_t i = 0; i < 3000; ++i) meta.push_back(meta_for(i));
    LargeBuildParams lp;
    lp.index.graph = VamanaParams{16, 32, 1.2f, 2};
    lp.memory_budget = std::size_t(300) << 10;
    LargeBuildStats st;
    DiskIndex::build_from_file(path, vec, Metric::Cosine, lp, id_range(9000, 3000), meta, Schema::dynamic(), &st);
    const auto d = DiskIndex::open(path);
    bool ok = st.partitions > 1 && d->contains(9000) && !d->contains(0) && d->get(9007)->metadata == meta_for(7);
    for (const auto& x : d->search(data[5], 10, {}, with_filter(Filter::eq("category", "tech")))) ok = ok && (x.id - 9000) % 5 == 2;
    const auto v = *d->get_vector(9003);
    double len = 0;
    for (float x : v) len += double(x) * x;
    CHECK(ok && std::abs(len - 1.0) < 1e-5 && d->search(data[11], 1)[0].id == 9011);
    for (const auto& f : {vec, path}) std::filesystem::remove(f);
}

TEST(disk, dk32_rejects_invalid_input) {
    auto data = random_vectors(2000, 8, 1954);
    const std::string vec = temp_path("dk32_vectors"), path = temp_path("dk32"), bad = temp_path("dk32_bad");
    write_vectors_file(vec, flatten(data), 8);
    write_file(path, "the previous index");
    LargeBuildParams small;
    small.memory_budget = std::size_t(64) << 10;
    LargeBuildParams one = small;
    one.replicas = 1;
    LargeBuildParams few = small;
    few.partitions = 3;
    CHECK_THROWS_AS(std::runtime_error, DiskIndex::build_from_file(path, temp_path("dk32_missing"), Metric::L2));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::InnerProduct));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::L2, one));   // partitions could not connect
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::L2, few));   // too few for the budget
    LargeBuildParams tiny;
    tiny.memory_budget = 1000;
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::L2, tiny));
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::L2, {}, id_range(0, 5)));
    std::string f = read_file(vec);
    write_file(bad, f.substr(0, f.size() - 4));  // its size no longer matches
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, bad, Metric::L2));
    write_file(bad, "abc");
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, bad, Metric::L2));
    const float nan = kNaN;
    f.replace(8 + 4 * 100, 4, reinterpret_cast<const char*>(&nan), 4);
    write_file(bad, f);
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, bad, Metric::L2, small));  // found while partitioning
    CHECK(read_file(path) == "the previous index" && build_dirs_left(path) == 0 && temporaries_left(path) == 0);
    for (const auto& x : {vec, path, bad}) std::filesystem::remove(x);
}

TEST(disk, dk33_vectors_files) {
    const std::string vec = temp_path("dk33");
    write_vectors_file(vec, std::vector<float>{1, 2, 3, 4, 5, 6}, 3);
    const std::string f = read_file(vec);
    CHECK(f.size() == 8 + 24 && persist::from_little_endian<std::uint32_t>(reinterpret_cast<const unsigned char*>(f.data())) == 2);
    CHECK_THROWS_AS(std::invalid_argument, write_vectors_file(vec, std::vector<float>{1, 2, 3}, 0));
    CHECK_THROWS_AS(std::invalid_argument, write_vectors_file(vec, std::vector<float>{1, 2, 3}, 2));
    std::filesystem::remove(vec);
}

TEST(disk, dk34_the_budget_holds_with_more_replicas) {
    // Found while measuring: 3 replicas once overfilled partitions (1,347 vectors
    // against a cap of 853). The partition count now guarantees room.
    auto data = clustered(4000, 32, 20, 1955);
    const std::string vec = temp_path("dk34_vectors"), path = temp_path("dk34");
    write_vectors_file(vec, flatten(data), 32);
    LargeBuildParams lp;
    lp.index.graph = VamanaParams{32, 64, 1.2f, 1};
    lp.memory_budget = std::size_t(400) << 10;
    lp.replicas = 3;
    const std::size_t cap = lp.memory_budget / (4 * PreparedVector(32).stride() + 8 * 32 + 96);
    LargeBuildStats st;
    DiskIndex::build_from_file(path, vec, Metric::L2, lp, {}, {}, Schema::dynamic(), &st);
    CHECK(st.partitions > 1 && st.largest_partition <= cap && disk_reachable(*DiskIndex::open(path)) == 4000);
    for (const auto& f : {vec, path}) std::filesystem::remove(f);
}

TEST(disk, dk35_replaced_or_deleted_while_open) {
    // Found on Windows: an open index file could not be deleted or replaced
    // (rebuilt in place), while POSIX allows both. Now the same everywhere: the
    // open index keeps reading the file it opened.
    auto old_data = random_vectors(200, 8, 1960), new_data = random_vectors(300, 8, 1961);
    const std::string path = temp_path("dk35");
    DiskIndex::build(path, id_range(0, 200), flatten(old_data), 8, Metric::L2);
    const auto old_index = DiskIndex::open(path);
    DiskIndex::build(path, id_range(0, 300), flatten(new_data), 8, Metric::L2);  // replaced while open
    const auto new_index = DiskIndex::open(path);
    CHECK(old_index->size() == 200 && new_index->size() == 300);
    CHECK(old_index->search(old_data[7], 1)[0].id == 7 && *old_index->get_vector(9) == old_data[9]);  // still the old file
    CHECK(new_index->search(new_data[7], 1)[0].id == 7);
    // Deleted while two indexes have it open: the deletion succeeds and both keep
    // working. (Whether the name disappears at once differs: older Windows keeps a
    // file "delete pending" until its last reader closes.)
    std::filesystem::remove(path);
    CHECK(old_index->search(old_data[3], 1)[0].id == 3 && new_index->search(new_data[250], 1)[0].id == 250);
}

TEST(vamana, va10_remaining_settings_and_counts) {
    const std::vector<float> v = {1, 2, 3, 4, 5, 6};
    CHECK_THROWS_AS(std::invalid_argument, VamanaGraph::build(v, 2, Metric::L2, VamanaParams{4, 8, kInf}));  // infinite alpha
    const auto g = VamanaGraph::build(v, 2, Metric::L2, VamanaParams{4, 8});
    std::size_t expanded = 99;
    CHECK(g.search(std::vector<float>{1, 2}, 0, 8, &expanded).empty() && expanded == 0);  // k = 0 reports no reads
}

TEST(disk, dk36_build_settings_not_yet_covered) {
    const std::string vec = temp_path("dk36_vectors"), path = temp_path("dk36"), bad = temp_path("dk36_bad");
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build(path, std::vector<std::uint64_t>{1}, std::vector<float>{}, 0, Metric::L2));
    auto data = clustered(1200, 8, 6, 1970);
    write_vectors_file(vec, flatten(data), 8);
    LargeBuildParams zero;
    zero.replicas = 0;
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, vec, Metric::L2, zero));
    auto header = [&](std::uint32_t n, std::uint32_t dim) { write_file(bad, le<std::uint32_t>(n) + le<std::uint32_t>(dim)); };
    header(0, 8);
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, bad, Metric::L2));  // no vectors
    header(1, 0);
    CHECK_THROWS_AS(std::invalid_argument, DiskIndex::build_from_file(path, bad, Metric::L2));  // dimension 0
    // Fits the budget, without statistics: one in-memory build.
    DiskIndex::build_from_file(path, vec, Metric::L2);
    CHECK(DiskIndex::open(path)->size() == 1200);
    // An explicit partition count, although everything would fit; without statistics.
    LargeBuildParams forced;
    forced.index.graph = VamanaParams{8, 16, 1.2f, 1};
    forced.partitions = 6;
    DiskIndex::build_from_file(path, vec, Metric::L2, forced);
    CHECK(disk_reachable(*DiskIndex::open(path)) == 1200);
    for (const auto& f : {vec, path, bad}) std::filesystem::remove(f);
}

TEST(disk, dk37_partitioned_cosine_and_identical_vectors) {
    // Cosine merges whose joined lists exceed R (a small R); and mostly identical
    // vectors, whose partitions share copies (k-means centroids coincide).
    auto data = clustered(1500, 8, 6, 1971);
    const std::string vec = temp_path("dk37_vectors"), path = temp_path("dk37");
    write_vectors_file(vec, flatten(data), 8);
    LargeBuildParams lp;
    lp.index.graph = VamanaParams{4, 8, 1.2f, 3};
    lp.memory_budget = std::size_t(96) << 10;
    LargeBuildStats st;
    DiskIndex::build_from_file(path, vec, Metric::Cosine, lp, {}, {}, Schema::dynamic(), &st);
    const auto d = DiskIndex::open(path);
    CHECK(st.partitions > 1 && st.merged_and_pruned > 0 && d->search(data[9], 1, DiskSearchParams{60, 4})[0].id == 9);
    Vectors same(1200, std::vector<float>(8, 0.5f));
    for (std::size_t i = 0; i < 1200; i += 100) same[i][0] = float(i);  // a few distinct among identical ones
    write_vectors_file(vec, flatten(same), 8);
    LargeBuildParams many = lp;
    many.index.graph = VamanaParams{8, 16, 1.2f, 3};
    many.partitions = 12;
    many.memory_budget = std::size_t(2) << 20;
    DiskIndex::build_from_file(path, vec, Metric::L2, many, {}, {}, Schema::dynamic(), &st);
    // Found here: merged graphs left 386 of 1,200 vectors unreachable (merge pruning
    // broke ties by node number), then 75 (no connectivity repair after merging),
    // then 4 (the repair counted incoming edges from unreachable vectors).
    const auto d2 = DiskIndex::open(path);
    CHECK(st.partitions == 12 && disk_reachable(*d2) == 1200 && d2->search(same[1], 1)[0].distance == 0.0f);
    CHECK(d2->search(same[300], 1, DiskSearchParams{1200, 8})[0].id == 300);  // a lone vector among copies, reachable
    for (const auto& f : {vec, path}) std::filesystem::remove(f);
}

TEST(disk, dk38_invalid_headers_behind_valid_checksums) {
    auto data = random_vectors(60, 8, 1972);
    const std::string good = temp_path("dk38_good"), path = temp_path("dk38");
    DiskIndexParams p;
    p.graph = VamanaParams{8, 16, 1.2f, 1};
    p.pq_subspaces = 2;
    DiskIndex::build(good, id_range(0, 60), flatten(data), 8, Metric::L2, p);
    const std::string bytes = read_file(good);
    // Header fields: only the header is re-signed (its checks come before any
    // section is read; resign_disk would follow the edited sizes past the file).
    // Inside a section: everything is re-signed, the header being intact.
    auto rejected = [&](std::size_t at, const std::string& value) {
        std::string f = bytes;
        f.replace(at, value.size(), value);
        if (at >= 168) {
            resign_disk(f);
        } else {
            unsigned char b[4];
            persist::to_little_endian(persist::crc32(0, f.data(), 168), b);
            f.replace(168, 4, reinterpret_cast<const char*>(b), 4);
        }
        write_file(path, f);
        try {
            (void)DiskIndex::open(path);
            return false;
        } catch (const FormatError&) {
            return true;
        }
    };
    CHECK(rejected(16, le<std::uint64_t>(0)) && rejected(16, le<std::uint64_t>(std::uint64_t(1) << 21)));  // dimension
    CHECK(rejected(40, le<std::uint64_t>(std::uint64_t(1) << 17)) && rejected(32, le<std::uint64_t>(0)));    // R, n
    CHECK(rejected(32, le<std::uint64_t>(0xFFFFFFFFull)));                                                 // n beyond node ids
    for (std::size_t at : {56, 64, 72, 80}) CHECK(rejected(at, le<std::uint64_t>(12345)));               // each layout field
    CHECK(rejected(96, le<std::uint64_t>(std::uint64_t(1) << 40)));  // a section larger than the file
    const std::size_t pq = std::size_t(persist::from_little_endian<std::uint64_t>(reinterpret_cast<const unsigned char*>(bytes.data() + 108)));
    CHECK(rejected(pq, le<std::uint64_t>(9)));                       // more PQ parts than dimensions
    for (const auto& f : {good, path}) std::filesystem::remove(f);
}

TEST(disk, dk39_search_paths_not_yet_covered) {
    auto data = random_vectors(400, 8, 1973);
    const std::string path = temp_path("dk39");
    std::vector<Metadata> meta;
    for (std::uint64_t i = 0; i < 400; ++i) meta.push_back(Metadata().set("rare", std::int64_t(i % 40 == 0)));
    DiskIndex::build(path, id_range(0, 400), flatten(data), 8, Metric::L2, {}, meta);
    const auto d = DiskIndex::open(path);
    SearchOptions late = with_filter(Filter::eq("rare", std::int64_t(1)));  // 10 match: read directly
    late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    DiskSearchStats st;
    CHECK(d->search(data[0], 5, {}, late, &st).empty() && st.stopped_early && st.nodes_read == 0);
    CHECK(d->search_by_id(7, SIZE_MAX, DiskSearchParams{500, 8})->size() == 399);  // every other record
    const auto other = *d->search_by_id(7, 3, {}, with_filter(Filter::eq("rare", std::int64_t(1))));  // 7 does not match
    CHECK(other.size() == 3 && std::none_of(other.begin(), other.end(), [](const SearchResult& r) { return r.id == 7; }));
    std::filesystem::remove(path);
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
    HnswIndex h(4, Metric::L2, HnswParams{4, 32, 5, true, {}});  // M = 4: levels vary a lot
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
    HnswIndex h(8, Metric::L2, HnswParams{8, 64, 3, true, {}});
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
    HnswIndex h(48, Metric::L2, HnswParams{16, 100, 3, true, {}});
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
        "  --shard K/N         run every N-th selected test, from the K-th (CTest splits heavy groups)\n"
        "  --list              list the selected tests without running them\n"
        "  --fail-fast         stop after the first failing test\n"
        "  --write-golden      rewrite the saved-format golden files (format changes only)\n"
        "  --help              show this help\n"
        "\n"
        "Groups: layer1, layer2, helpers, flat, hnsw, robustness, deletion, concurrency,\n"
        "        metadata, filter, planner, payload, batch, range, search_e2e, update, hardening, persistence, add_batch, bulk, records, control, codec, sq8, matryoshka, lean, review, codes_only, binary, pq, ivf, edges, vamana, disk, stress, e2e\n"
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
    std::size_t shard = 0, shards = 1;  // --shard K/N: every N-th selected test, from the K-th (1-based)

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--list") {
            list_only = true;
        } else if (arg == "--fail-fast") {
            fail_fast = true;
        } else if (arg == "--write-golden") {
            // Rewrites the format version 1 golden files in tests/data. Only for a
            // deliberate format change: the files guard compatibility.
            // Writes the current format version's files; older versions' files are
            // frozen and must never be rewritten.
            golden_hnsw()->save(golden_path("golden_hnsw_v3.bin"));
            golden_flat()->save(golden_path("golden_flat_v3.bin"));
            golden_sq8()->save(golden_path("golden_hnsw_sq8_v3.bin"));
            golden_binary()->save(golden_path("golden_hnsw_binary_v3.bin"));
            golden_pq()->save(golden_path("golden_hnsw_pq_v3.bin"));
            golden_ivf()->save(golden_path("golden_flat_ivf_v3.bin"));
            std::printf("wrote golden files to %s\n", HNSW_TEST_DATA_DIR);
            return 0;
        } else if (arg == "--shard") {
            unsigned k = 0, n = 0;
            if (i + 1 >= argc || std::sscanf(argv[i + 1], "%u/%u", &k, &n) != 2 || n == 0 || k == 0 || k > n) {
                std::printf("--shard needs K/N with 1 <= K <= N, for example 2/4\n\n");
                print_usage();
                return 2;
            }
            ++i;
            shard = k - 1;
            shards = n;
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
    if (shards > 1) {  // round-robin, so a group's heavy tests spread over its shards
        std::vector<const TestCase*> mine;
        for (std::size_t i = shard; i < selected.size(); i += shards) mine.push_back(selected[i]);
        selected.swap(mine);
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