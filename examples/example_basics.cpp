// The basics: create an index, add vectors, search, remove, compact.
#include <cstdio>
#include <vector>

#include "hnsw_index.h"

int main() {
    // An approximate index for 4-dimensional vectors, compared by cosine distance.
    vecdb::HnswIndex index(4, vecdb::Metric::Cosine);

    // Add vectors under your own 64-bit IDs. Cosine vectors are normalized for you.
    index.add(100, std::vector<float>{0.9f, 0.1f, 0.0f, 0.0f});
    index.add(101, std::vector<float>{0.8f, 0.2f, 0.1f, 0.0f});
    index.add(102, std::vector<float>{0.0f, 0.1f, 0.9f, 0.2f});
    index.add(103, std::vector<float>{0.1f, 0.0f, 0.8f, 0.3f});

    // The 2 nearest neighbors of a query, closest first.
    const std::vector<float> query{1.0f, 0.15f, 0.05f, 0.0f};
    for (const vecdb::SearchResult& r : index.search(query, 2))
        std::printf("id %llu  distance %.4f\n", (unsigned long long)r.id, r.distance);

    // Remove a vector; its ID can be used again right away.
    index.remove(100);
    std::printf("after removing 100, the closest is %llu\n", (unsigned long long)index.search(query, 1)[0].id);

    // After many removals, compact() rebuilds the index without the removed vectors.
    const vecdb::CompactStats stats = index.compact();
    std::printf("compacted: %zu kept, %zu reclaimed\n", stats.kept, stats.reclaimed);
}