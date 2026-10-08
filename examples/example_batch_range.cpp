// Metadata and filtered search: "the nearest articles about tech from 2020 or later".
#include <cstdio>
#include <vector>

#include "hnsw_index.h"

int main() {
    using namespace vecdb;

    // A strict schema rejects unknown fields and wrong types (catches typos early).
    Schema schema = Schema::strict({{"topic", FieldType::Keyword},
                                    {"year", FieldType::Int},
                                    {"tags", FieldType::Tags}});
    HnswIndex index(3, Metric::L2, HnswParams{}, schema);

    index.add(1, std::vector<float>{0.1f, 0.2f, 0.3f}, Metadata().set("topic", "tech").set("year", 2021).set_tags("tags", {"ai"}));
    index.add(2, std::vector<float>{0.1f, 0.2f, 0.4f}, Metadata().set("topic", "tech").set("year", 2015));
    index.add(3, std::vector<float>{0.2f, 0.2f, 0.3f}, Metadata().set("topic", "sports").set("year", 2022));
    index.add(4, std::vector<float>{0.9f, 0.9f, 0.9f}, Metadata().set("topic", "tech").set("year", 2023).set_tags("tags", {"chips"}));

    SearchOptions options;
    options.filter = Filter::eq("topic", "tech") && Filter::ge("year", 2020);
    SearchStats stats;
    options.stats = &stats;  // optional: what the search did

    const std::vector<float> query{0.1f, 0.2f, 0.3f};
    for (const SearchResult& r : index.search(query, 10, 64, options))
        std::printf("id %llu  distance %.4f\n", (unsigned long long)r.id, r.distance);
    std::printf("strategy: %s, about %zu matching vectors\n",
                stats.strategy == Strategy::ForceExact ? "exact scan" : "graph search", stats.estimated_matches);

    // Conditions outside the index: any function of the user ID (here: odd IDs only).
    SearchOptions allowed;
    allowed.predicate = [](std::uint64_t id) { return id % 2 == 1; };
    std::printf("nearest allowed: %llu\n", (unsigned long long)index.search(query, 1, 64, allowed)[0].id);

    // Change one field without touching the vector.
    index.set_metadata(2, Metadata().set("year", 2024));
    std::printf("vector 2 is now from %lld\n", (long long)index.get_metadata(2)->get("year")->i);
}