// Builds against the packaged library and checks a search gives the right answer.
#include <cstdio>
#include <vector>

#include "hnsw_index.h"

int main() {
    vecdb::HnswIndex index(3, vecdb::Metric::L2);
    index.add(1, std::vector<float>{0.0f, 0.0f, 0.0f});
    index.add(2, std::vector<float>{1.0f, 1.0f, 1.0f});
    const auto found = index.search(std::vector<float>{0.9f, 1.0f, 1.1f}, 1);
    std::printf("hnsw-lite package: nearest id %llu\n", found.empty() ? 0ULL : (unsigned long long)found[0].id);
    return found.size() == 1 && found[0].id == 2 ? 0 : 1;
}
