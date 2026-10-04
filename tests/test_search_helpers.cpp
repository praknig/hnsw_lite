/**
 * @file test_search_helpers.cpp
 * @brief Tests for the Layer 3 helpers. Prints "All search helper tests passed."
 *
 * Checks that:
 *  - TopK keeps exactly the k best candidates and returns them sorted.
 *  - PreparedVector pads with zeros, aligns, normalizes only for cosine, and
 *    rejects wrong dimensions.
 *  - VisitedList tracks visits per search and survives epoch wrap-around.
 *  - VisitedListPool reuses lists and gets them back even after an exception.
 */
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "prepared_vector.h"
#include "search_result.h"
#include "visited_list.h"
#include "check.h"

using namespace vecdb;

static void test_topk() {
    TopK top(3);
    CHECK(top.size() == 0 && !top.full());
    CHECK(std::isinf(top.worst_distance()));

    const float distances[] = {5, 1, 4, 2, 3, 0.5f, 9};
    for (NodeId i = 0; i < 7; ++i) top.push({distances[i], i});
    CHECK(top.full() && top.size() == 3);
    CHECK(top.worst_distance() == 2.0f);
    CHECK(!top.push({7.0f, 99}));  // worse than everything kept

    auto best = top.take_sorted();
    CHECK(best.size() == 3);
    CHECK(best[0].id == 5 && best[1].id == 1 && best[2].id == 3);  // 0.5, 1, 2
    CHECK(top.size() == 0);  // emptied

    TopK ties(2);  // equal distances: smaller id wins
    ties.push({1.0f, 8});
    ties.push({1.0f, 3});
    ties.push({1.0f, 5});
    auto t = ties.take_sorted();
    CHECK(t[0].id == 3 && t[1].id == 5);

    TopK none(0);
    CHECK(!none.push({1.0f, 1}) && none.take_sorted().empty());
}

static void test_prepared_vector() {
    PreparedVector p(5);
    CHECK(p.stride() == 16 && p.dim() == 5);
    CHECK(reinterpret_cast<std::uintptr_t>(p.data()) % kAlign == 0);

    std::vector<float> v{3, 4, 0, 0, 0};
    p.prepare(v, Metric::L2);  // L2: copied unchanged
    CHECK(p.values()[0] == 3.0f && p.values()[1] == 4.0f);

    p.prepare(v, Metric::Cosine);  // cosine: scaled to length 1
    CHECK(std::fabs(p.values()[0] - 0.6f) < 1e-6f && std::fabs(p.values()[1] - 0.8f) < 1e-6f);

    std::vector<float> big{9, 9, 9, 9, 9};
    p.prepare(big, Metric::InnerProduct);  // reuse: padding must still be zero
    for (std::size_t i = 5; i < p.padded().size(); ++i) CHECK(p.padded()[i] == 0.0f);

    CHECK(throws([&] { p.prepare(std::vector<float>(4), Metric::L2); }));
    CHECK(throws([] { PreparedVector bad(0); }));
}

static void test_visited_list() {
    VisitedList v;
    v.reset(10);
    CHECK(v.visit(3));   // first visit
    CHECK(!v.visit(3));  // second visit in the same search
    CHECK(v.visited(3) && !v.visited(4));

    v.reset(10);  // new search: everything unvisited again
    CHECK(!v.visited(3) && v.visit(3));

    v.reset(20);  // grows when nodes were added
    CHECK(v.visit(19));

    v.set_epoch_for_testing(0xFFFFFFFFu);  // next reset wraps to 0
    v.visit(7);                            // mark 7 with the max epoch
    v.reset(20);                           // wrap: array cleared, epoch 1
    CHECK(!v.visited(7) && !v.visited(19));
    CHECK(v.visit(7));
}

static void test_visited_list_pool() {
    VisitedListPool pool;
    CHECK(pool.idle_count() == 0);
    {
        auto a = pool.acquire(10);
        auto b = pool.acquire(10);  // two at once: a second list is created
        CHECK(a->visit(1) && b->visit(1));  // independent lists
        CHECK(pool.idle_count() == 0);
    }
    CHECK(pool.idle_count() == 2);  // both returned automatically

    {
        auto c = pool.acquire(10);  // reused, and reset for a new search
        CHECK(pool.idle_count() == 1);
        CHECK(!c->visited(1));
    }

    try {
        auto d = pool.acquire(10);
        throw std::runtime_error("search failed");
    } catch (const std::runtime_error&) {
    }
    CHECK(pool.idle_count() == 2);  // returned even after an exception
}

int main() {
    test_topk();
    test_prepared_vector();
    test_visited_list();
    test_visited_list_pool();
    std::printf("All search helper tests passed.\n");
}