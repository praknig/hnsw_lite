/**
 * @file test_adversarial.cpp
 * @brief Tries to break hnsw-lite: absurd sizes, memory leaks and unbounded growth,
 * hostile numbers, structural abuse, corrupt files and concurrent misuse.
 *
 * Separate from test_comprehensive.cpp, which checks that the library works; this
 * checks what a careless caller or an attacker can do to it. Every test asserts a
 * precise outcome: a clear exception, a bounded result, unchanged state, or memory
 * that does not grow. Run it under the sanitizers (CI's address-sanitizer job does,
 * with leak detection), where any overflow, leak or undefined behavior fails it.
 *
 *     test_adversarial               every test
 *     test_adversarial magnitude     the tests whose names contain "magnitude"
 *
 * Memory and open-file measurements use /proc and run on Linux only; elsewhere
 * those checks are skipped (the operations still run, for the sanitizers).
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <dirent.h>
#endif

#include "collection.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "sparse.h"

using namespace vecdb;

// ---------------------------------------------------------------- a small harness

namespace {

struct Case {
    const char* name;
    void (*fn)();
};
std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}
struct Register {
    Register(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};
int g_failures = 0;
const char* g_current = "";

void fail(int line, const std::string& what) {
    ++g_failures;
    std::printf("    FAILED line %d: %s\n", line, what.c_str());
}

#define ADV_TEST(name)                                  \
    void adv_##name();                                  \
    const Register reg_##name(#name, adv_##name);       \
    void adv_##name()
#define CHECK(cond) \
    do {            \
        if (!(cond)) fail(__LINE__, #cond); \
    } while (0)
#define CHECK_THROWS(type, expr)                                                              \
    do {                                                                                      \
        bool thrown_ = false;                                                                 \
        try {                                                                                 \
            (void)(expr);                                                                     \
        } catch (const type&) {                                                               \
            thrown_ = true;                                                                   \
        } catch (const std::exception& e_) {                                                  \
            fail(__LINE__, std::string("threw another exception: ") + e_.what() + " from " #expr); \
            thrown_ = true;                                                                   \
        }                                                                                     \
        if (!thrown_) fail(__LINE__, "did not throw " #type ": " #expr);                      \
    } while (0)

constexpr std::size_t kHuge = std::numeric_limits<std::size_t>::max();

std::vector<std::vector<float>> random_vectors(std::size_t n, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0, 1);
    std::vector<std::vector<float>> out(n, std::vector<float>(dim));
    for (auto& v : out)
        for (float& x : v) x = d(rng);
    return out;
}

/// Resident memory in KB (Linux), or -1.
long rss_kb() {
#if defined(__linux__)
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        long pages = 0, resident = 0;
        const int n = std::fscanf(f, "%ld %ld", &pages, &resident);
        std::fclose(f);
        if (n == 2) return resident * 4;
    }
#endif
    return -1;
}

/// Open file descriptors (Linux), or -1.
int open_files() {
#if defined(__linux__)
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (readdir(d)) ++n;
        closedir(d);
        return n;
    }
#endif
    return -1;
}

/// Runs `round` `rounds` times and checks that resident memory does not KEEP growing.
/// Memory is measured for the whole process, which also reflects how the allocator
/// reuses or returns what earlier tests freed: a one-time shift can show up in any
/// single window (found on CI's coverage job, where b4 grew 7.9 MB once, then stayed
/// flat). A leak grows steadily, so the test measures two consecutive windows and fails
/// only if the second still grows by more than half of `slack_kb`. Skipped where
/// memory cannot be measured, or under AddressSanitizer, whose quarantine holds freed
/// memory on purpose (there, LeakSanitizer checks for leaks instead).
void check_no_growth(int rounds, long slack_kb, const std::function<void()>& round) {
    for (int i = 0; i < 3; ++i) round();  // warm-up
    const long before = rss_kb();
    for (int i = 0; i < rounds / 2; ++i) round();
    const long middle = rss_kb();
    for (int i = rounds / 2; i < rounds; ++i) round();
    const long after = rss_kb();
#if defined(__SANITIZE_ADDRESS__)
    constexpr bool asan = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    constexpr bool asan = true;
#else
    constexpr bool asan = false;
#endif
#else
    constexpr bool asan = false;
#endif
    if (before < 0 || asan) return;
    std::printf("    resident memory %ld KB -> %ld KB -> %ld KB over two windows of %d rounds\n", before, middle, after, rounds / 2);
    if (after - middle > slack_kb / 2)
        fail(__LINE__, "memory still growing in the second window: " + std::to_string(after - middle) + " KB");
}

std::string temp_dir(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / (std::string("hnsw_lite_adv_") + name);
    std::filesystem::remove_all(p);
    return p.string();
}

}  // namespace

// ================================================================ A. Absurd sizes

ADV_TEST(a1_absurd_k_ef_and_limits) {
    // Results are bounded by what is stored, never by what is asked: no buffer of
    // 2^64 entries is reserved (under AddressSanitizer that would abort at once).
    auto data = random_vectors(300, 8, 1);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    SparseIndex s;
    for (std::uint64_t i = 0; i < 300; ++i) {
        h.add(i, data[i]);
        f.add(i, data[i]);
        s.add(i, {{std::uint32_t(i % 7)}, {1.0f}});
    }
    CHECK(h.search(data[0], kHuge, 64).size() == 300);
    CHECK(h.search(data[0], 10, kHuge).size() == 10);
    CHECK(h.search_range(data[0], 1e30f, kHuge).size() == 300);
    CHECK(f.search(data[0], kHuge).size() == 300);
    CHECK(s.search({{1}, {1.0f}}, kHuge).size() <= 300);
    CHECK(h.search(data[0], 0, 64).empty());
    std::vector<float> batch(8 * 3);
    std::copy(data[0].begin(), data[0].end(), batch.begin());
    CHECK(h.search_batch(batch, kHuge, 64).size() == 3);
    Collection c(CollectionOptions::preset(8));
    for (std::uint64_t i = 0; i < 50; ++i) {
        Document d;
        d.vector = data[i];
        c.add(i, d);
    }
    CollectionQuery q;
    q.vector = data[0];
    CHECK(c.search(q, kHuge).size() == 50);
}

ADV_TEST(a2_absurd_dimensions) {
    // Refused before any allocation: a dimension near 2^64 would otherwise wrap a
    // size computation around to a tiny buffer.
    for (const std::size_t dim : {std::size_t(0), kMaxDim + 1, kHuge, kHuge / 4 + 1}) {
        CHECK_THROWS(std::invalid_argument, HnswIndex(dim, Metric::L2));
        CHECK_THROWS(std::invalid_argument, FlatIndex(dim, Metric::L2));
    }
    HnswIndex h(4, Metric::L2);
    CHECK_THROWS(std::invalid_argument, h.add(1, std::vector<float>(5, 1.0f)));
    CHECK_THROWS(std::invalid_argument, h.add(1, std::vector<float>{}));
    CHECK_THROWS(std::invalid_argument, (void)h.search(std::vector<float>(3, 1.0f), 5));
    CHECK_THROWS(std::invalid_argument, (void)h.search_batch(std::vector<float>(7, 1.0f), 5));  // not a multiple of 4
    CHECK(h.size() == 0);
}

ADV_TEST(a3_huge_metadata) {
    // Megabyte strings, thousands of fields and tags: stored and read back exactly,
    // and filters over them still work.
    HnswIndex h(4, Metric::L2);
    const std::string big(1 << 20, 'k');
    Metadata many;
    for (int i = 0; i < 3000; ++i) many.set("f" + std::to_string(i), std::int64_t(i));
    std::vector<std::string> tags;
    for (int i = 0; i < 10000; ++i) tags.push_back("t" + std::to_string(i));
    h.add(1, std::vector<float>{1, 0, 0, 0}, Metadata().set("big", big));
    h.add(2, std::vector<float>{0, 1, 0, 0}, many);
    h.add(3, std::vector<float>{0, 0, 1, 0}, Metadata().set_tags("tags", tags));
    CHECK(h.get_metadata(1)->get("big")->s == big);
    CHECK(h.get_metadata(2)->get("f2999")->i == 2999);
    CHECK(h.count_where(Filter::eq("big", big)) == 1);
    CHECK(h.count_where(Filter::has_tag("tags", "t9999")) == 1);
    CHECK(h.count_where(Filter::eq("f1500", std::int64_t(1500))) == 1);
}

// ================================================================ B. Leaks and growth

ADV_TEST(b1_create_and_destroy_everything) {
    // Every kind of index and collection, built and destroyed many times: LeakSanitizer
    // reports any allocation left behind; resident memory must not keep growing.
    auto data = random_vectors(200, 16, 2);
    check_no_growth(60, 4096, [&] {
        HnswIndex h(16, Metric::Cosine);
        FlatIndex f(16, Metric::InnerProduct);
        SparseIndex s;
        for (std::uint64_t i = 0; i < 200; ++i) {
            h.add(i, data[i], Metadata().set("g", std::int64_t(i % 5)).set("k", "v" + std::to_string(i)));
            f.add(i, data[i]);
            s.add(i, {{std::uint32_t(i % 11), std::uint32_t(100 + i % 3)}, {1.0f, 0.5f}});
        }
        (void)h.remove(5);
        (void)h.compact();
        (void)h.reorder();
        CollectionOptions o = CollectionOptions::preset(16);
        o.maintenance.mode = Maintenance::Manual;
        o.ttl.enabled = true;
        o.string_ids = true;
        Collection c(o);
        for (std::uint64_t i = 0; i < 50; ++i) {
            Document d;
            d.vector = data[i];
            d.text = "word" + std::to_string(i % 4) + " common";
            c.upsert("key" + std::to_string(i), d);
        }
        (void)c.maintain();
    });
}

ADV_TEST(b2_rejected_operations_leak_nothing) {
    // Thousands of operations that throw: each must leave nothing allocated and the
    // index unchanged.
    auto data = random_vectors(50, 8, 3);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) h.add(i, data[i]);
    std::ostringstream before;
    h.save(before);
    std::vector<float> nan_v(8, 1.0f);
    nan_v[3] = std::numeric_limits<float>::quiet_NaN();
    check_no_growth(300, 2048, [&] {
        CHECK_THROWS(std::invalid_argument, h.add(1000, nan_v));
        CHECK_THROWS(std::invalid_argument, h.add(1, data[1]));  // a duplicate id
        CHECK_THROWS(std::invalid_argument, (void)h.search(nan_v, 5));
        CHECK_THROWS(std::invalid_argument, (void)h.search(std::vector<float>(3, 1.0f), 5));
    });
    std::ostringstream after;
    h.save(after);
    CHECK(before.str() == after.str());
}

ADV_TEST(b3_durable_open_close_releases_files) {
    const std::string dir = temp_dir("b3");
    std::vector<float> v(8, 0.5f);
    const int before = open_files();
    for (int i = 0; i < 150; ++i) {
        auto d = HnswIndex::open_durable(dir, 8, Metric::L2);
        d->add(std::uint64_t(i), v);
        if (i % 50 == 0) (void)d->checkpoint();
    }
    for (int i = 0; i < 50; ++i) {
        CollectionOptions o = CollectionOptions::preset(8);
        o.maintenance.mode = Maintenance::Manual;
        auto c = Collection::open(dir + "_c", o);
        Document doc;
        doc.vector = v;
        (void)c->upsert(std::uint64_t(i), doc);
    }
    const int after = open_files();
    if (before >= 0) {
        std::printf("    open files: %d before, %d after 200 open/close cycles\n", before, after);
        CHECK(after == before);
    }
    auto again = HnswIndex::open_durable(dir, 8, Metric::L2);
    CHECK(again->size() == 150);
    again.reset();
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(dir + "_c");
}

ADV_TEST(b4_unique_values_do_not_accumulate) {
    // Every write introduces new keyword values, replacing old ones: the value
    // dictionary must forget values no document holds any more (with maintenance).
    std::vector<float> v(8, 0.5f);
    CollectionOptions o = CollectionOptions::preset(8);
    o.maintenance.mode = Maintenance::Manual;
    Collection c(o);
    int round = 0;
    check_no_growth(25, 6144, [&] {
        for (std::uint64_t i = 0; i < 1000; ++i) {
            Document d;
            d.vector = v;
            d.metadata = Metadata().set("k", "value-" + std::to_string(round) + "-" + std::to_string(i) + std::string(200, 'x'));
            (void)c.upsert(i, d);
        }
        (void)c.maintain();
        ++round;
    });
    CHECK(c.size() == 1000);
}

ADV_TEST(b5_save_and_load_cycles) {
    auto data = random_vectors(300, 16, 4);
    HnswParams p;
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 100;
    HnswIndex h(16, Metric::L2, p);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i], Metadata().set("g", std::int64_t(i % 3)));
    std::ostringstream first;
    h.save(first);
    check_no_growth(80, 4096, [&] {
        std::istringstream in(first.str());
        auto copy = HnswIndex::load(in);
        std::ostringstream out;
        copy->save(out);
        CHECK(out.str() == first.str());
    });
}

// ================================================================ C. Hostile numbers

ADV_TEST(c1_nan_and_infinity_everywhere) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    SparseIndex s;
    CollectionOptions o = CollectionOptions::preset(4);
    o.maintenance.mode = Maintenance::Manual;
    Collection c(o);
    h.add(1, std::vector<float>{1, 2, 3, 4});
    for (const float bad : {nan, inf, -inf}) {
        const std::vector<float> v{1, bad, 3, 4};
        CHECK_THROWS(std::invalid_argument, h.add(2, v));
        CHECK_THROWS(std::invalid_argument, f.add(2, v));
        CHECK_THROWS(std::invalid_argument, (void)h.search(v, 3));
        CHECK_THROWS(std::invalid_argument, (void)h.search_batch(v, 3));
        CHECK_THROWS(std::invalid_argument, s.add(2, {{1}, {bad}}));
        Document d;
        d.vector = v;
        CHECK_THROWS(std::invalid_argument, c.add(2, d));
    }
    CHECK(h.size() == 1 && f.size() == 0 && s.size() == 0 && c.size() == 0);
}

ADV_TEST(c2_magnitudes_that_overflow_distances) {
    // The library accepts any finite float, by design (see the robustness group in
    // test_comprehensive.cpp): a distance that overflows becomes infinity, never NaN,
    // and infinities sort last. Attacked here with values near the float maximum on
    // every metric and every search path.
    const std::vector<float> huge(4, 3e38f), other{-3e38f, 3e38f, -3e38f, 3e38f};
    for (const Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        HnswIndex h(4, m);
        FlatIndex f(4, m);
        auto data = random_vectors(100, 4, 9);
        for (std::uint64_t i = 0; i < 100; ++i) {
            h.add(i, data[i]);
            f.add(i, data[i]);
        }
        h.add(500, huge);
        h.add(501, other);
        f.add(500, huge);
        f.add(501, other);
        bool no_nan = true, sorted = true;
        for (const auto& query : {huge, other, data[0]}) {
            for (const auto& r : {h.search(query, 20, 64), f.search(query, 20), h.search_range(query, 1e30f, 200)}) {
                for (std::size_t i = 0; i < r.size(); ++i) {
                    no_nan = no_nan && !std::isnan(r[i].distance);
                    if (i > 0) sorted = sorted && !(r[i].distance < r[i - 1].distance);
                }
            }
        }
        CHECK(no_nan && sorted);
        // Exact search always finds the vector itself first (its distance to itself is
        // finite: 0 for L2, a valid value for the others).
        CHECK(f.search(huge, 1)[0].id == 500 || m == Metric::InnerProduct);
        SearchOptions exact;
        exact.strategy = Strategy::ForceExact;
        CHECK(h.search(huge, 1, 64, exact)[0].id == 500 || m == Metric::InnerProduct);
        if (m == Metric::Cosine) CHECK(h.search(huge, 1)[0].id == 500);  // normalized: an ordinary unit vector
    }
    // A documented limitation, not a crash: in an L2 graph, a vector whose distances
    // to every neighbor overflow ties them all at infinity, so the greedy walk has no
    // direction to follow toward it; exact search (above) still finds it.
    HnswIndex g(4, Metric::L2);
    auto data = random_vectors(200, 4, 10);
    for (std::uint64_t i = 0; i < 200; ++i) g.add(i, data[i]);
    g.add(999, huge);
    const auto r = g.search(huge, 3);
    CHECK(r.size() == 3 && !std::isnan(r[0].distance));
    std::printf("    L2 graph search for a 3e38 vector found %s (exact search finds it)\n", r[0].id == 999 ? "it" : "a neighbor at infinity");
}

ADV_TEST(c3_zero_tiny_and_identical_vectors) {
    // A zero vector under cosine: stored as zeros (no NaN), distance 1 to everything.
    HnswIndex cosine(4, Metric::Cosine);
    cosine.add(1, std::vector<float>{1, 0, 0, 0});
    cosine.add(2, std::vector<float>{0, 0, 0, 0});
    const auto z = *cosine.get_vector(2);
    CHECK(std::all_of(z.begin(), z.end(), [](float x) { return x == 0.0f; }));
    const auto zr = cosine.search(std::vector<float>{0, 0, 0, 0}, 2);
    CHECK(zr.size() == 2 && std::isfinite(zr[0].distance) && std::isfinite(zr[1].distance));
    // Denormal values: stored exactly and found.
    HnswIndex tiny(4, Metric::L2);
    const float d = std::numeric_limits<float>::denorm_min();
    tiny.add(1, std::vector<float>{d, 0, 0, 0});
    tiny.add(2, std::vector<float>{0, d * 4, 0, 0});
    CHECK(tiny.search(std::vector<float>{d, 0, 0, 0}, 1)[0].id == 1);
    // 2,000 identical vectors: every distance ties; searches still terminate with k
    // results, and removing most of them keeps the rest findable.
    HnswIndex same(8, Metric::L2);
    const std::vector<float> v(8, 0.25f);
    for (std::uint64_t i = 0; i < 2000; ++i) same.add(i, v);
    CHECK(same.search(v, 10).size() == 10);
    for (std::uint64_t i = 0; i < 1990; ++i) (void)same.remove(i);
    const auto rest = same.search(v, 20, 200);
    CHECK(rest.size() == 10);
}

// ================================================================ D. Structural abuse

ADV_TEST(d1_deep_and_wide_filters) {
    HnswIndex h(4, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) h.add(i, std::vector<float>{float(i), 0, 0, 0}, Metadata().set("g", std::int64_t(i % 10)));
    // Deep: refused at a fixed depth, with no stack overflow while building or destroying.
    bool refused = false;
    try {
        Filter f = Filter::eq("g", std::int64_t(1));
        for (int i = 0; i < 100000; ++i) f = f && Filter::exists("g");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
    // Wide: a 100,000-value IN list and a 50,000-tag condition work.
    std::vector<Value> values;
    for (std::int64_t i = 0; i < 100000; ++i) values.push_back(Value::of_int(i * 10 + 3));
    CHECK(h.count_where(Filter::in("g", values)) == 10);
    std::vector<std::string> tags;
    for (int i = 0; i < 50000; ++i) tags.push_back("tag" + std::to_string(i));
    CHECK_THROWS(std::invalid_argument, (void)h.count_where(Filter::has_any_tag("g", tags)));  // an integer field: refused
    HnswIndex t(4, Metric::L2);
    for (std::uint64_t i = 0; i < 20; ++i) t.add(i, std::vector<float>{float(i), 0, 0, 0}, Metadata().set_tags("tags", {"tag" + std::to_string(i * 1000)}));
    CHECK(t.count_where(Filter::has_any_tag("tags", tags)) == 20 && t.count_where(Filter::has_all_tags("tags", tags)) == 0);
}

ADV_TEST(d2_operations_on_empty_and_emptied_indexes) {
    HnswIndex h(8, Metric::L2);
    const std::vector<float> v(8, 1.0f);
    CHECK(h.search(v, 5).empty() && h.search_range(v, 10.0f, 5).empty());
    (void)h.compact();
    (void)h.reorder();
    (void)h.upkeep();
    HealthOptions ho;
    CHECK(h.health(ho).unreachable == 0);
    auto data = random_vectors(300, 8, 5);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i]);
    for (std::uint64_t i = 0; i < 300; ++i) CHECK(h.remove(i));
    CHECK(h.size() == 0 && h.search(v, 5).empty());
    (void)h.compact();
    (void)h.reorder();
    (void)h.upkeep();
    h.add(7, v);  // and it still works
    CHECK(h.search(v, 1)[0].id == 7);
}

ADV_TEST(d3_writes_from_inside_a_search) {
    // A predicate is user code running inside a search: reads of the same index are
    // allowed, writes are refused (they would deadlock or corrupt the walk).
    HnswIndex h(4, Metric::L2);
    for (std::uint64_t i = 0; i < 50; ++i) h.add(i, std::vector<float>{float(i), 1, 0, 0});
    int refused = 0, reads = 0;
    SearchOptions o;
    o.predicate = [&](std::uint64_t id) {
        try {
            h.add(1000 + id, std::vector<float>{1, 1, 1, 1});
        } catch (const std::logic_error&) {
            ++refused;
        }
        try {
            (void)h.remove(id);
        } catch (const std::logic_error&) {
            ++refused;
        }
        reads += h.contains(id) ? 1 : 0;
        return true;
    };
    const auto r = h.search(std::vector<float>{0, 1, 0, 0}, 5, 32, o);
    CHECK(r.size() == 5 && refused > 0 && reads > 0 && h.size() == 50);
}

// ================================================================ E. Corrupt files

ADV_TEST(e1_every_truncation_and_byte_flips) {
    // Every loader, given every prefix of a valid file and files with single bytes
    // flipped: a clean exception or (for flips the format does not cover) a valid
    // object, never a crash, a leak or undefined behavior.
    auto data = random_vectors(40, 4, 6);
    HnswIndex h(4, Metric::L2);
    FlatIndex f(4, Metric::L2);
    SparseIndex s;
    CollectionOptions o = CollectionOptions::preset(4);
    o.maintenance.mode = Maintenance::Manual;
    Collection c(o);
    for (std::uint64_t i = 0; i < 40; ++i) {
        h.add(i, data[i], Metadata().set("k", "v" + std::to_string(i % 3)));
        f.add(i, data[i]);
        s.add(i, {{std::uint32_t(i % 5)}, {1.0f}});
        Document d;
        d.vector = data[i];
        d.text = "some words " + std::to_string(i);
        c.add(i, d);
    }
    struct Loader {
        const char* name;
        std::string bytes;
        std::function<void(std::istream&)> load;
    };
    std::ostringstream hs, fs, ss, cs;
    h.save(hs);
    f.save(fs);
    s.save(ss);
    c.save(cs);
    std::vector<Loader> loaders = {{"hnsw", hs.str(), [](std::istream& in) { (void)HnswIndex::load(in); }},
                                   {"flat", fs.str(), [](std::istream& in) { (void)FlatIndex::load(in); }},
                                   {"sparse", ss.str(), [](std::istream& in) { (void)SparseIndex::load(in); }},
                                   {"collection", cs.str(), [](std::istream& in) { (void)Collection::load(in); }}};
    std::mt19937 rng(7);
    for (const Loader& l : loaders) {
        std::size_t refused = 0, tried = 0;
        auto attempt = [&](const std::string& bytes) {
            ++tried;
            std::istringstream in(bytes);
            try {
                l.load(in);
            } catch (const std::exception&) {  // FormatError and friends: clean refusals
                ++refused;
            }
        };
        const std::size_t step = std::max<std::size_t>(1, l.bytes.size() / 400);  // at most ~400 truncations
        for (std::size_t n = 0; n < l.bytes.size(); n += step) attempt(l.bytes.substr(0, n));
        for (int i = 0; i < 300; ++i) {
            std::string b = l.bytes;
            b[rng() % b.size()] ^= static_cast<char>(1 + rng() % 255);
            attempt(b);
        }
        std::printf("    %-10s %zu damaged files, %zu refused cleanly, none crashed\n", l.name, tried, refused);
        CHECK(refused > 0);
    }
}

// ================================================================ F. Concurrency abuse

ADV_TEST(f1_every_operation_at_once) {
    // Eight threads mixing every kind of operation on one index for a while. The
    // index's lock must keep it consistent: no crash, no unexpected exception, a
    // healthy graph afterwards, and every remaining id findable by its own vector.
    auto data = random_vectors(600, 8, 8);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i], Metadata().set("g", std::int64_t(i % 4)));
    std::atomic<bool> stop{false};
    std::atomic<int> unexpected{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            std::mt19937 rng(100 + t);
            while (!stop) {
                const std::uint64_t id = rng() % 600;
                try {
                    switch (rng() % 10) {
                        case 0: (void)h.upsert(id, data[id], Metadata().set("g", std::int64_t(id % 4))); break;
                        case 1: (void)h.remove(id); break;
                        case 2: {
                            SearchOptions o;
                            o.filter = Filter::eq("g", std::int64_t(id % 4));
                            (void)h.search(data[id], 5, 32, o);
                            break;
                        }
                        case 3: (void)h.search_range(data[id], 2.0f, 20); break;
                        case 4: {
                            std::ostringstream out;
                            h.save(out);
                            break;
                        }
                        case 5: if (t == 0) (void)h.compact(); break;
                        case 6: if (t == 1) (void)h.upkeep(); break;
                        case 7: if (t == 2) (void)h.reorder(); break;
                        default: (void)h.search(data[id], 10); break;
                    }
                } catch (const std::exception& e) {
                    ++unexpected;
                    std::printf("    unexpected: %s\n", e.what());
                }
            }
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    stop = true;
    for (auto& t : threads) t.join();
    CHECK(unexpected == 0);
    (void)h.upkeep();
    HealthOptions ho;
    ho.recall_sample = 0;
    CHECK(h.health(ho).unreachable == 0);
    bool findable = true;
    for (std::uint64_t i = 0; i < 600; ++i)
        if (h.contains(i)) findable = findable && h.search(data[i], 1, 64)[0].id == i;
    CHECK(findable);
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    std::printf("hnsw-lite adversarial tests\n");
    int run = 0;
    for (const Case& c : cases()) {
        if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
        g_current = c.name;
        const int before = g_failures;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
        } catch (const std::exception& e) {
            fail(0, std::string("unexpected exception: ") + e.what());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %s  %-44s %8.1f ms\n", g_failures == before ? "PASS" : "FAIL", c.name, ms);
        ++run;
    }
    std::printf("%d tests, %d failed checks\n", run, g_failures);
    return g_failures == 0 && run > 0 ? 0 : 1;
}