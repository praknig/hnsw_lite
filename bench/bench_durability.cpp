/**
 * @file bench_durability.cpp
 * @brief Additions per second without durability and in each durability mode.
 *
 * Usage: bench_durability [count] [directory]   (default 2000, the temp directory)
 * The flush cost depends entirely on the storage: run it on the disk you use.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>

#include "hnsw_index.h"

using namespace vecdb;

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::size_t(std::atol(argv[1])) : 2000, dim = 64;
    const std::filesystem::path base = argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::temp_directory_path();
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    std::vector<std::vector<float>> data(n, std::vector<float>(dim));
    for (auto& v : data)
        for (float& x : v) x = nd(rng);
    auto run = [&](const char* name, auto&& make) {
        auto index = make();
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < n; ++i) index->add(i, data[i]);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("%-12s %9.0f additions/s  (%6.1f us each)\n", name, double(n) / s, 1e6 * s / double(n));
    };
    std::printf("%zu additions of %zu dimensions, in %s\n\n", n, dim, base.string().c_str());
    run("not durable", [&] { return std::make_unique<HnswIndex>(dim, Metric::L2); });
    for (auto [mode, name] : {std::pair{Durability::Buffered, "buffered"}, std::pair{Durability::Grouped, "grouped"},
                              std::pair{Durability::Synced, "synced"}}) {
        const std::filesystem::path dir = base / ("hnsw_lite_bench_" + std::string(name));
        std::filesystem::remove_all(dir);
        DurableOptions o;
        o.durability = mode;
        run(name, [&] { return HnswIndex::open_durable(dir.string(), dim, Metric::L2, {}, Schema::dynamic(), o); });
        std::filesystem::remove_all(dir);
    }
}
