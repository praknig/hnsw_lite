// Tests for Layer 1. Build and run:
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Iinclude tests/test_layer1.cpp -o test && ./test
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "../include/storage.h"

using namespace vecdb;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAILED line %d: %s\n", __LINE__, #cond);            \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static bool aligned(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kAlign == 0; }

template <class F>
static bool throws(F f) {
    try { f(); } catch (...) { return true; }
    return false;
}

static void test_aligned_block() {
    AlignedBlock b(100);
    CHECK(aligned(b.data()));
    CHECK(b.size() == 128);  // rounded up to whole cache lines
    for (std::size_t i = 0; i < b.size(); ++i) CHECK(b.data()[i] == std::byte{0});
    AlignedBlock moved = std::move(b);
    CHECK(b.data() == nullptr && moved.data() != nullptr);
}

static void test_arena() {
    Arena a(1024);
    auto* x = a.allocate_array<std::uint64_t>(10);
    auto* y = a.allocate_array<std::uint64_t>(10);
    CHECK(reinterpret_cast<std::uintptr_t>(x) % alignof(std::uint64_t) == 0);
    CHECK(y >= x + 10);                       // no overlap
    a.allocate(5000);                         // bigger than a block
    CHECK(a.block_count() == 2);
}

static void test_vector_store() {
    VectorStore s(100, /*shelf_bits=*/2);  // 4 vectors per shelf, to test spilling
    CHECK(s.stride() == 112);

    std::vector<float> v(100);
    std::iota(v.begin(), v.end(), 1.0f);
    NodeId id0 = s.add(v);
    const float* first_address = s.get(id0).data();

    for (int i = 0; i < 10; ++i) s.add(v);  // fills 3 shelves
    CHECK(s.size() == 11);
    CHECK(s.get(id0).data() == first_address);  // nothing ever moved

    for (NodeId id = 0; id < s.size(); ++id) {
        CHECK(aligned(s.get(id).data()));
        CHECK(s.get(id)[99] == 100.0f);
        auto padded = s.get_padded(id);
        for (std::size_t i = 100; i < padded.size(); ++i) CHECK(padded[i] == 0.0f);
    }
    CHECK(throws([&] { s.get(99); }));
    CHECK(throws([&] { s.add(std::vector<float>(5)); }));
}

static void test_id_map() {
    IdMap m;
    CHECK(m.add(5000) == 0);
    CHECK(m.add(42) == 1);
    CHECK(throws([&] { m.add(42); }));
    CHECK(*m.find(42) == 1 && !m.find(7));
    CHECK(m.external(0) == 5000);
    m.mark_deleted(1);
    CHECK(m.is_deleted(1) && !m.is_deleted(0));
}

static void test_graph_storage() {
    GraphStorage g(16, /*shelf_bits=*/2);
    NodeId a = g.add_node(0);
    NodeId b = g.add_node(2);
    for (int i = 0; i < 6; ++i) g.add_node(0);  // spill onto new shelves

    CHECK(g.links(a, 0).size() == 32 && g.links(b, 2).size() == 16);
    CHECK(aligned(g.links(a, 0).data()));
    CHECK(GraphStorage::count(g.links(a, 0)) == 0);  // starts empty
    CHECK(throws([&] { g.links(a, 1); }));           // a only lives on level 0

    g.links(a, 0)[0] = b;
    g.links(a, 0)[1] = 7;
    g.links(b, 2)[0] = a;
    CHECK(GraphStorage::count(g.links(a, 0)) == 2);
    CHECK(g.links(b, 2)[0] == a);
    CHECK(GraphStorage::count(g.links(b, 1)) == 0);  // level 1 untouched
}

static void test_storage() {
    Storage st(3);
    std::vector<float> v{1, 2, 3};
    NodeId id = st.insert(777, v, 1);
    CHECK(st.vectors().get(id)[2] == 3.0f);
    CHECK(st.graph().level(id) == 1);
    CHECK(*st.ids().find(777) == id);
    CHECK(throws([&] { st.insert(777, v, 0); }));
    CHECK(st.size() == 1);  // failed insert changed nothing
}

int main() {
    test_aligned_block();
    test_arena();
    test_vector_store();
    test_id_map();
    test_graph_storage();
    test_storage();
    std::printf("All Layer 1 tests passed.\n");
}