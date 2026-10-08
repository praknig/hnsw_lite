// Batch search (many queries in one call) and range search (everything within a distance).
#include <cstdio>
#include <vector>

#include "flat_index.h"

int main() {
    using namespace vecdb;
    FlatIndex index(2, Metric::L2);  // exact search
    for (int i = 0; i < 10; ++i) index.add(std::uint64_t(i), std::vector<float>{float(i), 0.0f});

    // Batch: queries stored back to back (2 floats each), answered on 4 threads.
    const std::vector<float> queries{0.2f, 0.0f,   // query 0
                                     8.9f, 0.0f};  // query 1
    const auto batch = index.search_batch(queries, 2, 4);
    for (std::size_t q = 0; q < batch.size(); ++q)
        std::printf("query %zu: ids %llu and %llu\n", q, (unsigned long long)batch[q][0].id, (unsigned long long)batch[q][1].id);

    // Range: every vector within distance 2.5 of (5, 0). L2 compares squared
    // distances, so convert the radius with l2_radius().
    const std::vector<float> center{5.0f, 0.0f};
    std::printf("within 2.5:");
    for (const SearchResult& r : index.search_range(center, l2_radius(2.5f)))
        std::printf(" %llu", (unsigned long long)r.id);
    std::printf("\n");
}