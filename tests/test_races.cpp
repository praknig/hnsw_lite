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
#include "disk_index.h"
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

// ---------------------------------------------------------------- r10

RACE_TEST(r10_disk_index_shared_by_many_searchers) {
    // One file, two open instances, many threads searching both, with and without a
    // filter: concurrent results must equal the sequential ones (search is
    // deterministic), whatever the interleaving.
    const std::string path = temp_dir("r10") + ".disk";
    auto data = random_vectors(800, 8, 10);
    std::vector<std::uint64_t> ids(800);
    std::vector<float> flat;
    std::vector<Metadata> md;
    for (std::uint64_t i = 0; i < 800; ++i) {
        ids[i] = i;
        flat.insert(flat.end(), data[i].begin(), data[i].end());
        md.push_back(Metadata().set("g", std::int64_t(i % 4)));
    }
    DiskIndex::build(path, ids, flat, 8, Metric::L2, {}, md);
    auto a = DiskIndex::open(path);
    auto b = DiskIndex::open(path);
    SearchOptions filtered;
    filtered.filter = Filter::eq("g", std::int64_t(2));
    std::vector<std::vector<std::uint64_t>> expected(40), expected_f(40);
    for (std::size_t q = 0; q < 40; ++q) {
        for (const auto& r : a->search(data[q * 7], 10)) expected[q].push_back(r.id);
        for (const auto& r : a->search(data[q * 7], 10, {}, filtered)) expected_f[q].push_back(r.id);
    }
    std::atomic<int> mismatches{0};
    auto searcher = [&](const DiskIndex& d) {
        return [&](std::size_t, std::mt19937& rng) {
            const std::size_t q = rng() % 40;
            std::vector<std::uint64_t> got, got_f;
            for (const auto& r : d.search(data[q * 7], 10)) got.push_back(r.id);
            for (const auto& r : d.search(data[q * 7], 10, {}, filtered)) got_f.push_back(r.id);
            if (got != expected[q] || got_f != expected_f[q]) ++mismatches;
        };
    };
    collide({searcher(*a), searcher(*a), searcher(*a), searcher(*b), searcher(*b)});
    CHECK(mismatches == 0);
    a.reset();
    b.reset();
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------- r11

RACE_TEST(r11_bulk_changes_against_keyed_writes) {
    // remove_where and set_metadata_where sweep the whole collection while keyed
    // writes, reads and filtered counts run: every count stays within bounds and
    // every document read back is complete.
    auto data = random_vectors(300, 8, 11);
    CollectionOptions o = manual_options(8);
    o.string_ids = true;
    Collection c(o);
    std::atomic<int> bad{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {
            const std::size_t i = rng() % 300;
            Document d;
            d.vector = data[i];
            d.metadata = Metadata().set("g", std::int64_t(i % 5)).set("flag", false);
            (void)c.upsert("k" + std::to_string(i), d);
        },
        [&](std::size_t, std::mt19937& rng) {
            if (rng() % 2) (void)c.set_metadata_where(Filter::eq("g", std::int64_t(rng() % 5)), Metadata().set("flag", true));
            else (void)c.remove_where(Filter::eq("g", std::int64_t(rng() % 5)) && Filter::eq("flag", true));
        },
        [&](std::size_t, std::mt19937& rng) {
            const std::size_t i = rng() % 300;
            if (const auto r = c.get("k" + std::to_string(i)))
                if (!r->metadata.get("g") || r->metadata.get("g")->i != std::int64_t(i % 5) || !r->metadata.get("flag")) ++bad;
            // Each count bounded on its own: two calls are two moments, and writers run in
            // between, so "some <= all" across them would not hold (found on macOS CI).
            const std::size_t all = c.count_where(Filter::all()), some = c.count_where(Filter::eq("flag", true));
            if (all > 300 || some > 300) ++bad;
        },
        [&](std::size_t, std::mt19937& rng) {
            CollectionQuery q;
            q.vector = data[rng() % 300];
            CollectionSearchOptions so;
            so.filter = Filter::eq("flag", false);
            so.with_metadata = true;
            for (const auto& h : c.search(q, 5, so))
                if (!h.metadata || !h.metadata->get("flag") || h.metadata->get("flag")->b) ++bad;
        },
    });
    CHECK(bad == 0 && c.count_where(Filter::all()) == c.size());
}

// ---------------------------------------------------------------- r12

RACE_TEST(r12_snapshots_taken_during_writes) {
    // Saving while writers run: every snapshot must load back internally consistent
    // (counts agree, every stored document readable and findable by its own vector).
    auto data = random_vectors(300, 8, 12);
    Collection c(manual_options(8));
    FlatIndex f(8, Metric::L2);
    for (std::uint64_t i = 0; i < 100; ++i) {
        Document d;
        d.vector = data[i];
        d.text = "word" + std::to_string(i % 3);
        c.add(i, d);
        f.add(i, data[i]);
    }
    std::atomic<int> inconsistent{0}, snapshots{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t i = rng() % 300;
            Document d;
            d.vector = data[i];
            d.text = "word" + std::to_string(i % 3);
            (void)c.upsert(i, d);
            if (rng() % 3 == 0) (void)c.remove(rng() % 300);
            (void)f.upsert(i, data[i]);
            if (rng() % 3 == 0) (void)f.remove(rng() % 300);
        },
        [&](std::size_t, std::mt19937&) {
            std::ostringstream out;
            c.save(out);
            std::istringstream in(out.str());
            auto copy = Collection::load(in);
            if (copy->count_where(Filter::all()) != copy->size()) ++inconsistent;
            for (std::uint64_t i = 0; i < 300; ++i) {
                const auto r = copy->get(i);
                if (!r) continue;
                if (!r->vector || *r->vector != data[i]) {
                    ++inconsistent;
                    continue;
                }
                CollectionQuery q;
                q.vector = *r->vector;
                const auto hits = copy->search(q, 1);
                if (hits.empty() || hits[0].id != i) ++inconsistent;
            }
            ++snapshots;
        },
        [&](std::size_t, std::mt19937&) {
            std::ostringstream out;
            f.save(out);
            std::istringstream in(out.str());
            auto copy = FlatIndex::load(in);
            for (std::uint64_t i = 0; i < 300; ++i)
                if (const auto v = copy->get_vector(i); v && *v != data[i]) ++inconsistent;
        },
    });
    std::printf("    %d collection snapshots checked\n", snapshots.load());
    CHECK(inconsistent == 0 && snapshots > 0);
}

// ---------------------------------------------------------------- r13

RACE_TEST(r13_collection_scans_while_writing) {
    // Full scans, a page at a time, while writers add, update and remove documents:
    // always ascending, never a duplicate, never a half-written document, and every
    // document the writers never touch (IDs below 100) appears exactly once per scan.
    auto data = random_vectors(300, 8, 13);
    CollectionOptions o = manual_options(8);
    Collection c(o);
    auto doc = [&](std::size_t i) {
        Document d;
        d.vector = data[i];
        d.text = "word" + std::to_string(i % 3);
        d.metadata = Metadata().set("g", std::int64_t(i % 5));
        return d;
    };
    for (std::uint64_t i = 0; i < 100; ++i) c.add(i, doc(i));
    std::atomic<int> bad{0}, scans{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t i = 100 + rng() % 200;
            if (rng() % 3) (void)c.upsert(i, doc(i));
            else (void)c.remove(i);
        },
        [&](std::size_t, std::mt19937& rng) {
            CollectionScanOptions so;
            so.limit = 1 + rng() % 40;
            std::vector<std::uint64_t> ids;
            for (;;) {
                const CollectionPage page = c.scan(so);
                for (const auto& d : page.documents) {
                    ids.push_back(d.id);
                    const auto g = d.metadata.get("g");
                    if (!d.vector || d.vector->size() != 8 || !g || g->i != std::int64_t(d.id % 5) || d.text != "word" + std::to_string(d.id % 3)) ++bad;
                }
                if (!page.next) break;
                so.after = page.next;
            }
            if (!std::is_sorted(ids.begin(), ids.end()) || std::adjacent_find(ids.begin(), ids.end()) != ids.end()) ++bad;
            std::size_t stable = 0;
            for (const std::uint64_t id : ids) stable += id < 100;
            if (stable != 100) ++bad;
            ++scans;
        },
    });
    std::printf("    %d full scans checked\n", scans.load());
    CHECK(bad == 0 && scans > 0);
}

// ---------------------------------------------------------------- r14

RACE_TEST(r14_facets_while_indexed_metadata_changes) {
    // Facets (over everything, a filter, and given IDs) while writers change indexed
    // metadata: every result is consistent. A keyword field holds one value or none, so
    // the counts plus the documents without it always make the total.
    auto data = random_vectors(200, 8, 14);
    HnswIndex h(8, Metric::L2);
    Collection c(manual_options(8));
    const char* langs[] = {"en", "fr", "de"};
    auto md = [&](std::size_t i) { return i % 7 == 0 ? Metadata() : Metadata().set("lang", langs[i % 3]).set("tags", std::vector<std::string>{"t" + std::to_string(i % 4)}); };
    for (std::uint64_t i = 0; i < 100; ++i) {
        h.add(i, data[i], md(i + 1));
        Document d;
        d.vector = data[i];
        d.metadata = md(i + 1);
        c.add(i, d);
    }
    h.create_payload_index("lang");
    h.create_payload_index("tags");
    c.create_payload_index("lang");
    std::atomic<int> bad{0};
    auto consistent = [&](const FacetResult& r) {
        std::size_t sum = 0;
        for (const auto& v : r.values) sum += v.count;
        return sum + r.missing + r.other == r.total && r.total <= 200;
    };
    const std::vector<FacetRequest> req{FacetRequest{"lang", 0, {}, true}};
    collide({
        [&](std::size_t, std::mt19937& rng) {
            const std::uint64_t i = rng() % 200;
            if (rng() % 4 == 0) {
                (void)h.remove(i);
                (void)c.remove(i);
            } else {
                (void)h.upsert(i, data[i], md(rng()));
                Document d;
                d.vector = data[i];
                d.metadata = md(rng());
                (void)c.upsert(i, d);
            }
        },
        [&](std::size_t, std::mt19937& rng) {
            const Filter f = Filter::has_tag("tags", "t" + std::to_string(rng() % 4));
            for (const auto& r : {h.facets(req)[0], h.facets(req, f)[0], c.facets(req)[0], c.facets(req, f)[0]})
                if (!consistent(r)) ++bad;
        },
        [&](std::size_t, std::mt19937& rng) {
            std::vector<std::uint64_t> ids;
            for (int k = 0; k < 20; ++k) ids.push_back(rng() % 220);
            if (!consistent(h.facets_for(ids, req)[0]) || !consistent(c.facets_for(ids, req)[0])) ++bad;
        },
    });
    CHECK(bad == 0);
}

// ---------------------------------------------------------------- r15

RACE_TEST(r15_fallbacks_rabitq_compaction_and_half_disk) {
    // Filtered searches falling back to exact scans while writers change the filtered field
    // and a thread compacts a codes-only RaBitQ index (codes copied, not re-encoded), with
    // readers sharing a half-precision disk index. Every result respects its filter.
    auto data = random_vectors(400, 8, 15);
    QuantizationParams qp;
    qp.codec = CodecKind::RaBitQ;
    qp.train_after = 50;
    qp.keep_full_vectors = false;
    HnswParams hp;
    hp.quantization = qp;
    HnswIndex h(8, Metric::L2, hp);
    // "fixed" never changes (the ID's last digit): results are checked against it from the
    // ID alone, with no read-back race. "c" changes under every write.
    for (std::uint64_t i = 0; i < 300; ++i) h.add(i, data[i], Metadata().set("c", std::int64_t(i % 10)).set("fixed", std::int64_t(i % 10)));
    PlannerParams tight;
    tight.fallback_factor = 0.25;
    tight.max_exact_matches = 0;
    tight.acorn_below = 0;
    h.set_planner_params(tight);
    const std::string disk = (std::filesystem::temp_directory_path() / "r15_disk.idx").string();
    DiskIndexParams dp;
    dp.graph = VamanaParams{8, 16, 1.2f, 1};
    dp.vectors = DiskVectorFormat::Float16;
    std::vector<std::uint64_t> ids(400);
    std::vector<float> flat;
    for (std::uint64_t i = 0; i < 400; ++i) {
        ids[i] = i;
        flat.insert(flat.end(), data[i].begin(), data[i].end());
    }
    DiskIndex::build(disk, ids, flat, 8, Metric::L2, dp);
    const auto d = DiskIndex::open(disk);
    std::atomic<int> bad{0};
    collide({
        [&](std::size_t, std::mt19937& rng) {  // writers: the filtered field and the vectors change
            const std::uint64_t i = rng() % 400;
            if (rng() % 3 == 0) (void)h.remove(i);
            else (void)h.upsert(i, data[i], Metadata().set("c", std::int64_t(rng() % 10)).set("fixed", std::int64_t(i % 10)));
        },
        [&](std::size_t, std::mt19937& rng) {  // searches: correlated filters, many falling back
            SearchOptions o;
            const std::int64_t below = std::int64_t(1 + rng() % 4);
            const bool stable = rng() % 2 == 0;  // the stable field, or the one changing under writes
            o.filter = stable ? Filter::lt("fixed", below) : Filter::lt("c", below);
            for (const auto& r : h.search(data[rng() % 400], 10, 16, o)) {
                if (stable && std::int64_t(r.id % 10) >= below) ++bad;  // a result outside its filter
                if (std::isnan(r.distance) || r.id >= 400) ++bad;
            }
        },
        [&](std::size_t, std::mt19937&) { (void)h.compact(); },  // codes-only compaction, copying codes
        [&](std::size_t, std::mt19937& rng) {  // the half-precision disk index, shared
            for (const auto& r : d->search(data[rng() % 400], 5))
                if (std::isnan(r.distance) || r.id >= 400) ++bad;
        },
    });
    std::filesystem::remove(disk);
    CHECK(bad == 0);
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
