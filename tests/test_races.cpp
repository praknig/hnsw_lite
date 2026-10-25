/**
 * @file test_races.cpp
 * @brief Race-condition tests: threads deliberately colliding on every kind of
 * shared state in hnsw-lite.
 *
 * A race depends on timing, so a test can pass a thousand times and still hide
 * one. These tests are written for ThreadSanitizer, which reports any unsynchronized
 * access to shared memory even when the timing happened to work out (CI's
 * thread-sanitizer job runs them; locally: configure with
 * -DCMAKE_CXX_FLAGS=-fsanitize=thread). Without it, each test still checks that
 * the results stay consistent: no unexpected exception, a valid final state.
 *
 *     test_races              every test
 *     test_races r01          the tests whose names contain "r01"
 *
 * Each test runs its threads for a fixed time (RACE_MS, default 400 ms).
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "collection.h"
#include "flat_index.h"
#include "hnsw_index.h"
#include "sparse.h"

using namespace vecdb;

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
std::atomic<int> g_failures{0};

void fail(int line, const std::string& what) {
    ++g_failures;
    std::printf("    FAILED line %d: %s\n", line, what.c_str());
}

#define RACE_TEST(name)                             \
    void race_##name();                             \
    const Register reg_##name(#name, race_##name);  \
    void race_##name()
#define CHECK(cond) \
    do {            \
        if (!(cond)) fail(__LINE__, #cond); \
    } while (0)

std::chrono::milliseconds duration() {
    const char* s = std::getenv("RACE_MS");
    return std::chrono::milliseconds(s ? std::atoi(s) : 400);
}

std::vector<std::vector<float>> random_vectors(std::size_t n, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0, 1);
    std::vector<std::vector<float>> out(n, std::vector<float>(dim));
    for (auto& v : out)
        for (float& x : v) x = d(rng);
    return out;
}

/// Runs each worker in its own thread until the test's time is up. A worker is
/// called repeatedly with (its index, a per-thread random generator); an exception
/// it does not handle is counted as a failure.
void collide(const std::vector<std::function<void(std::size_t, std::mt19937&)>>& workers) {
    std::atomic<bool> stop{false};
    std::atomic<int> unexpected{0};
    std::vector<std::thread> threads;
    for (std::size_t w = 0; w < workers.size(); ++w)
        threads.emplace_back([&, w] {
            std::mt19937 rng(static_cast<std::uint32_t>(1000 + w));
            while (!stop.load(std::memory_order_relaxed)) {
                try {
                    workers[w](w, rng);
                } catch (const std::exception& e) {
                    if (++unexpected <= 3) std::printf("    unexpected in worker %zu: %s\n", w, e.what());
                }
            }
        });
    std::this_thread::sleep_for(duration());
    stop = true;
    for (auto& t : threads) t.join();
    if (unexpected) fail(__LINE__, std::to_string(unexpected.load()) + " unexpected exceptions");
}

std::string temp_dir(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / (std::string("hnsw_lite_race_") + name);
    std::filesystem::remove_all(p);
    return p.string();
}

CollectionOptions manual_options(std::size_t dim) {
    CollectionOptions o = CollectionOptions::preset(dim);
    o.maintenance.mode = Maintenance::Manual;
    return o;
}

}  // namespace

// ---------------------------------------------------------------- r01

RACE_TEST(r01_runtime_settings_while_keyed_operations_run) {
    // set_ttl() and set_string_ids() change settings that keyed operations, the
    // preparation of documents and remove_expired() read. Every reader must see
    // them under the collection's lock.
    auto data = random_vectors(200, 8, 1);
    CollectionOptions o = manual_options(8);
    o.ttl.enabled = true;
    o.string_ids = true;
    Collection c(o);
    std::atomic<int> expired_seen{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {  // keyed writes, some expiring at once
            const std::size_t i = rng() % 200;
            Document d;
            d.vector = data[i];
            if (rng() % 3 == 0) d.expires_at = now_ms() - 1;
            try {
                (void)c.upsert("k" + std::to_string(i), d);
            } catch (const std::invalid_argument&) {  // expiry switched off just now: expires_at refused
            } catch (const std::logic_error&) {  // string ids switched off just now: refused, as documented
            }
        },
        [&](std::size_t, std::mt19937& rng) {  // keyed reads and removals
            const std::string key = "k" + std::to_string(rng() % 200);
            try {
                if (const auto r = c.get(key); r && r->expires_at && *r->expires_at < now_ms() - 200) ++expired_seen;
                if (rng() % 7 == 0) (void)c.remove(key);
                (void)c.set_metadata(key, Metadata().set("n", std::int64_t(1)));
            } catch (const std::logic_error&) {
            }
        },
        [&](std::size_t, std::mt19937& rng) {  // searches and expiry
            CollectionQuery q;
            q.vector = data[rng() % 200];
            (void)c.search(q, 5);
            (void)c.remove_expired();
            (void)c.count_where(Filter::all());
        },
        [&](std::size_t, std::mt19937& rng) {  // the settings, flipping
            c.set_ttl(TtlOptions{rng() % 4 != 0, 0});
            c.set_string_ids(rng() % 4 != 0);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        },
    });
    c.set_ttl(TtlOptions{true, 0});
    c.set_string_ids(true);
    CHECK(expired_seen == 0);
}

// ---------------------------------------------------------------- r02

RACE_TEST(r02_background_maintenance_policy_changes_and_destruction) {
    // The maintenance thread runs while its policy changes, writers and searchers
    // continue, and finally the collection is destroyed in the middle of it all.
    auto data = random_vectors(300, 8, 2);
    for (int round = 0; round < 3; ++round) {
        CollectionOptions o = CollectionOptions::preset(8);
        o.maintenance.mode = Maintenance::Background;
        o.maintenance.interval = std::chrono::milliseconds(1);
        o.ttl.enabled = true;
        auto c = std::make_unique<Collection>(o);
        collide({
            [&](std::size_t, std::mt19937& rng) {
                const std::size_t i = rng() % 300;
                Document d;
                d.vector = data[i];
                d.text = "word" + std::to_string(i % 5);
                d.expires_at = now_ms() + std::int64_t(rng() % 20);
                (void)c->upsert(i, d);
                if (rng() % 4 == 0) (void)c->remove(rng() % 300);
            },
            [&](std::size_t, std::mt19937& rng) {
                CollectionQuery q;
                q.vector = data[rng() % 300];
                q.text = "word1";
                (void)c->search(q, 5);
            },
            [&](std::size_t, std::mt19937& rng) {
                MaintenancePolicy p = o.maintenance;
                p.mode = rng() % 2 ? Maintenance::Background : Maintenance::Manual;
                p.interval = std::chrono::milliseconds(1 + rng() % 3);
                c->set_maintenance(p);
                (void)c->maintenance_status();
            },
        });
        c.reset();  // destroyed while its maintenance thread may be mid-round
    }
}

// ---------------------------------------------------------------- r03

RACE_TEST(r03_durable_collection_writes_checkpoints_and_maintenance) {
    const std::string dir = temp_dir("r03");
    auto data = random_vectors(300, 8, 3);
    std::atomic<std::uint64_t> written{0};
    {
        auto c = Collection::open(dir, manual_options(8));
        collide({
            [&](std::size_t w, std::mt19937& rng) {
                const std::uint64_t id = (rng() % 150) * 2 + w % 2;
                Document d;
                d.vector = data[id];
                d.text = "durable text";
                (void)c->upsert(id, d);
                ++written;
            },
            [&](std::size_t w, std::mt19937& rng) {
                const std::uint64_t id = (rng() % 150) * 2 + w % 2;
                Document d;
                d.vector = data[id];
                (void)c->upsert(id, d);
                ++written;
            },
            [&](std::size_t, std::mt19937&) {
                (void)c->checkpoint();
                (void)c->maintain();
            },
            [&](std::size_t, std::mt19937& rng) {
                CollectionQuery q;
                q.vector = data[rng() % 300];
                (void)c->search(q, 5);
            },
        });
    }
    auto again = Collection::open(dir, manual_options(8));
    CHECK(again->size() > 0 && again->size() <= 300 && written > 0);
    again.reset();
    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------- r04

RACE_TEST(r04_index_setters_while_searching_and_writing) {
    auto data = random_vectors(400, 8, 4);
    HnswIndex h(8, Metric::L2);
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 200; ++i) {
        h.add(i, data[i], Metadata().set("g", std::int64_t(i % 8)));
        f.add(i, data[i], Metadata().set("g", std::int64_t(i % 8)));
    }
    collide({
        [&](std::size_t, std::mt19937& rng) {
            SearchOptions o;
            o.filter = Filter::eq("g", std::int64_t(rng() % 8));
            (void)h.search(data[rng() % 400], 5, 32, o);
            (void)f.search(data[rng() % 400], 5, o);
        },
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t id = 200 + rng() % 200;
            (void)h.upsert(id, data[id], Metadata().set("g", std::int64_t(id % 8)));
            if (rng() % 3 == 0) (void)h.remove(200 + rng() % 200);
        },
        [&](std::size_t, std::mt19937& rng) {
            PlannerParams p;
            p.max_ef_multiplier = 1.0 + double(rng() % 50);
            p.acorn_below = double(rng() % 100) / 100.0;
            h.set_planner_params(p);
            h.set_auto_compact(rng() % 2 ? 0.0 : 0.3);
        },
    });
    CHECK(h.search(data[0], 1)[0].id == 0);
}

// ---------------------------------------------------------------- r05

RACE_TEST(r05_the_shared_thread_pool) {
    // Parallel batch searches from several threads at once, parallel batch inserts,
    // and reorder() (which assigns on the pool too).
    auto data = random_vectors(2000, 8, 5);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1100; ++i) h.add(i, data[i]);
    std::vector<float> batch;
    for (std::size_t i = 0; i < 64; ++i) batch.insert(batch.end(), data[i].begin(), data[i].end());
    std::atomic<std::uint64_t> next{1100};
    collide({
        [&](std::size_t, std::mt19937&) { CHECK(h.search_batch(batch, 5, 32, 4).size() == 64); },
        [&](std::size_t, std::mt19937&) { CHECK(h.search_batch(batch, 5, 32, 3).size() == 64); },
        [&](std::size_t, std::mt19937&) {
            const std::uint64_t start = next.fetch_add(20);
            if (start + 20 > 2000) return;
            std::vector<std::uint64_t> ids(20);
            std::vector<float> vs;
            for (std::uint64_t i = 0; i < 20; ++i) {
                ids[i] = start + i;
                vs.insert(vs.end(), data[start + i].begin(), data[start + i].end());
            }
            h.add_batch(ids, vs, 4);
        },
        [&](std::size_t, std::mt19937&) {
            (void)h.reorder();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        },
    });
    CHECK(h.search(data[5], 1)[0].id == 5);
}

// ---------------------------------------------------------------- r06

RACE_TEST(r06_hybrid_search_while_both_sides_change) {
    auto data = random_vectors(300, 8, 6);
    HnswIndex dense(8, Metric::L2);
    SparseIndex sparse;
    for (std::uint64_t i = 0; i < 300; ++i) {
        dense.add(i, data[i], Metadata().set("gold", i % 10 == 0));
        sparse.add(i, {{std::uint32_t(i % 13), 50}, {1.0f, 0.5f}}, Metadata().set("gold", i % 10 == 0));
    }
    collide({
        [&](std::size_t, std::mt19937& rng) {
            HybridOptions o;
            if (rng() % 2) {
                o.modifiers = ScoreModifiers{};
                o.modifiers->boosts.push_back({Filter::eq("gold", true), 1.0});
            }
            (void)hybrid_search(dense, data[rng() % 300], sparse, {{std::uint32_t(rng() % 13)}, {1.0f}}, 5, o);
        },
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t id = rng() % 300;
            (void)dense.upsert(id, data[id], Metadata().set("gold", rng() % 2 == 0));
        },
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t id = rng() % 300;
            if (rng() % 2) (void)sparse.remove(id);
            else if (!sparse.contains(id)) sparse.add(id, {{std::uint32_t(id % 13)}, {1.0f}});
        },
    });
}

// ---------------------------------------------------------------- r07

RACE_TEST(r07_training_while_searching) {
    // IVF lists retrained, and a codec trained automatically by the insert that
    // reaches train_after, while searches run.
    auto data = random_vectors(3000, 8, 7);
    IvfParams ivf;
    ivf.lists = 8;
    ivf.probes = 3;
    ivf.train_after = 0;
    FlatIndex f(8, Metric::L2, Schema::dynamic(), {}, ivf);
    for (std::uint64_t i = 0; i < 400; ++i) f.add(i, data[i]);
    HnswParams p;
    p.quantization.codec = CodecKind::SQ8;
    p.quantization.train_after = 600;
    HnswIndex h(8, Metric::L2, p);
    std::atomic<std::uint64_t> next{0};
    collide({
        [&](std::size_t, std::mt19937&) {
            const std::uint64_t id = next.fetch_add(1);
            if (id >= 3000) return;
            h.add(id, data[id]);
            if (id >= 400) f.add(id, data[id]);
        },
        [&](std::size_t, std::mt19937& rng) {
            (void)h.search(data[rng() % 3000], 5);
            (void)f.search(data[rng() % 3000], 5);
        },
        [&](std::size_t, std::mt19937&) {
            f.train_lists();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        },
    });
    CHECK(h.size() > 0);
}

// ---------------------------------------------------------------- r08

RACE_TEST(r08_sparse_index_everything_at_once) {
    SparseIndex s;
    for (std::uint64_t i = 0; i < 200; ++i) s.add(i, {{std::uint32_t(i % 17), 100}, {1.0f, 0.25f}}, Metadata().set("g", std::int64_t(i % 3)));
    collide({
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t id = rng() % 400;
            if (!s.contains(id)) {
                try {
                    s.add(id, {{std::uint32_t(id % 17)}, {1.0f}});
                } catch (const std::invalid_argument&) {  // added by another thread meanwhile
                }
            }
        },
        [&](std::size_t, std::mt19937& rng) { (void)s.remove(rng() % 400); },
        [&](std::size_t, std::mt19937& rng) {
            SearchOptions o;
            o.filter = Filter::eq("g", std::int64_t(rng() % 3));
            (void)s.search({{std::uint32_t(rng() % 17)}, {1.0f}}, 5, o);
        },
        [&](std::size_t, std::mt19937&) {
            std::ostringstream out;
            s.save(out);
            std::istringstream in(out.str());
            CHECK(SparseIndex::load(in) != nullptr);
        },
    });
}

// ---------------------------------------------------------------- r09

RACE_TEST(r09_cancellation_and_deadlines_from_other_threads) {
    // Long operations cancelled at random moments by another thread: each either
    // completes or throws Cancelled, leaving a usable index; searches continue.
    auto data = random_vectors(1500, 8, 9);
    HnswIndex h(8, Metric::L2);
    for (std::uint64_t i = 0; i < 1500; ++i) h.add(i, data[i]);
    std::atomic<bool> cancel{false};
    std::atomic<int> cancelled{0}, completed{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {
            OperationOptions o;
            o.cancel = &cancel;
            cancel = false;
            try {
                switch (rng() % 3) {
                    case 0: (void)h.compact(o); break;
                    case 1: (void)h.reorder(o); break;
                    default: (void)h.rebuild({}, o); break;
                }
                ++completed;
            } catch (const Cancelled&) {
                ++cancelled;
            }
        },
        [&](std::size_t, std::mt19937& rng) {
            std::this_thread::sleep_for(std::chrono::microseconds(rng() % 3000));
            cancel = true;
        },
        [&](std::size_t, std::mt19937& rng) {
            SearchOptions o;
            o.deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(rng() % 500);
            (void)h.search(data[rng() % 1500], 5, 64, o);
        },
    });
    std::printf("    %d completed, %d cancelled\n", completed.load(), cancelled.load());
    CHECK(h.size() == 1500 && h.search(data[7], 1)[0].id == 7);
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    std::printf("hnsw-lite race-condition tests (%lld ms each)\n", static_cast<long long>(duration().count()));
    int run = 0;
    for (const Case& c : cases()) {
        if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
        const int before = g_failures;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
        } catch (const std::exception& e) {
            fail(0, std::string("unexpected exception: ") + e.what());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %s  %-58s %8.1f ms\n", g_failures == before ? "PASS" : "FAIL", c.name, ms);
        ++run;
    }
    std::printf("%d tests, %d failed checks\n", run, g_failures.load());
    return g_failures == 0 && run > 0 ? 0 : 1;
}