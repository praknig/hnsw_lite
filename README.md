# hnsw-lite

[![CI](https://github.com/praknig/hnsw_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/praknig/hnsw_lite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, fast, in-memory **vector search library** in modern C++20.

Store vectors (for example, text or image embeddings) under your own IDs, optionally with metadata, then find the vectors closest to a query: **exactly** with a Flat index, or **approximately and much faster** with an HNSW graph. Searches can be filtered by metadata, run in batches, or return everything within a distance.

hnsw-lite was built from scratch to show how libraries like hnswlib, Faiss and Qdrant work inside, so the code is meant to be read as well as used. It is tested thoroughly: 418 tests, 100% line coverage, memory and thread checkers, and CI on Linux, Windows and macOS.

## Features

- **Two indexes:** `FlatIndex` (exact, brute force) and `HnswIndex` (approximate, about 14x faster at 99.8% recall on 50,000 vectors).
- **Three distance metrics:** squared Euclidean (L2), inner product and cosine.
- **SIMD speed on any CPU:** AVX2, AVX-512 or ARM NEON versions are chosen automatically at runtime.
- **Real deletion:** removed IDs can be reused immediately, and `compact()` rebuilds a heavily changed index.
- **Metadata and filtered search:** integer, float, boolean, keyword and tag fields; filters like `Filter::eq("topic", "tech") && Filter::ge("year", 2020)`; a query planner that picks the fastest strategy.
- **Batch search** on a thread pool and **range search** (everything within a radius).
- **Safe by design:** invalid input is rejected with clear exceptions, and every change is all-or-nothing, even when memory runs out.

## Quick start

### 1. Build and run the tests

You need a C++20 compiler (GCC 11, Clang 14 or Visual Studio 2022 or newer) and CMake 3.20 or newer.

```bash
git clone https://github.com/praknig/hnsw_lite.git
cd hnsw_lite
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

On Windows with MinGW or w64devkit, add `-G Ninja -DCMAKE_CXX_COMPILER=g++` to the first `cmake` command. See [Getting started](docs/getting-started.md) for every platform, including Visual Studio and CLion.

### 2. Add it to your project

With CMake's FetchContent, hnsw-lite is downloaded and built as part of your project:

```cmake
include(FetchContent)
FetchContent_Declare(hnsw_lite GIT_REPOSITORY https://github.com/praknig/hnsw_lite.git GIT_TAG master)
FetchContent_MakeAvailable(hnsw_lite)

target_link_libraries(your_app PRIVATE hnsw_lite::hnsw_lite)
```

Or copy the repository into your project and use `add_subdirectory(hnsw_lite)`. Either way, only the library is built, not the tests.

### 3. Your first search

```cpp
#include <cstdio>
#include <vector>

#include "hnsw_index.h"

int main() {
    // An approximate index for 4-dimensional vectors, compared by cosine distance.
    vecdb::HnswIndex index(4, vecdb::Metric::Cosine);

    // Add vectors under your own 64-bit IDs.
    index.add(100, std::vector<float>{0.9f, 0.1f, 0.0f, 0.0f});
    index.add(101, std::vector<float>{0.8f, 0.2f, 0.1f, 0.0f});
    index.add(102, std::vector<float>{0.0f, 0.1f, 0.9f, 0.2f});
    index.add(103, std::vector<float>{0.1f, 0.0f, 0.8f, 0.3f});

    // The 2 nearest neighbors of a query, closest first.
    const std::vector<float> query{1.0f, 0.15f, 0.05f, 0.0f};
    for (const vecdb::SearchResult& r : index.search(query, 2))
        std::printf("id %llu  distance %.4f\n", (unsigned long long)r.id, r.distance);
}
```

Output:

```
id 100  distance 0.0020
id 101  distance 0.0071
```

## Examples

### Filter by metadata

```cpp
using namespace vecdb;
HnswIndex index(3, Metric::L2, HnswParams{},
                Schema::strict({{"topic", FieldType::Keyword}, {"year", FieldType::Int}}));

index.add(1, std::vector<float>{0.1f, 0.2f, 0.3f}, Metadata().set("topic", "tech").set("year", 2021));
index.add(2, std::vector<float>{0.1f, 0.2f, 0.4f}, Metadata().set("topic", "tech").set("year", 2015));
index.add(3, std::vector<float>{0.2f, 0.2f, 0.3f}, Metadata().set("topic", "sports").set("year", 2022));

SearchOptions options;
options.filter = Filter::eq("topic", "tech") && Filter::ge("year", 2020);
std::vector<float> query{0.1f, 0.2f, 0.3f};
auto results = index.search(query, /*k=*/10, /*ef=*/64, options);  // only vector 1 matches
```

### Batch and range search

```cpp
FlatIndex index(2, Metric::L2);
// ... add vectors ...

// Many queries in one call, stored back to back, on 4 threads.
std::vector<float> queries{0.2f, 0.0f,   8.9f, 0.0f};
auto per_query = index.search_batch(queries, /*k=*/2, /*threads=*/4);

// Everything within distance 2.5 of a point (L2 compares squared distances: use l2_radius).
std::vector<float> center{5.0f, 0.0f};
auto nearby = index.search_range(center, l2_radius(2.5f));
```

### Remove and compact

```cpp
index.remove(100);   // the ID can be added again immediately
index.compact();     // after many removals: rebuild without them
```

Complete, runnable versions of these are in [examples/](examples): `example_basics`, `example_filters` and `example_batch_range`. They are built with the project; run them from the `build` folder.

## Choosing an index and settings

| If you need... | Use |
|---|---|
| Exact results, or fewer than about 10,000 vectors | `FlatIndex` |
| Speed on larger collections, and can accept missing a few true neighbors | `HnswIndex` |
| Text embeddings from most models | `Metric::Cosine` |
| Raw coordinates or distances | `Metric::L2` |

For HNSW, the defaults (`M = 16`, `ef_construction = 200`) suit most data. At search time, `ef` (default 64) is the speed-versus-accuracy dial: raise it for better recall, lower it for speed. [Performance and tuning](docs/performance.md) has measured numbers and guidance.

## Things to know

- **Distances are "smaller is closer"** for every metric: L2 returns the *squared* distance, inner product returns the *negative* dot product, and cosine returns 1 minus the cosine similarity.
- **Thread safety:** any number of searches may run at once, but changes (`add`, `remove`, `set_metadata`, `compact`) must not overlap with anything else. Protect writes with your own lock for now.
- **Errors:** invalid input (wrong dimension, NaN, a duplicate ID, a metadata type mismatch) throws `std::invalid_argument` and leaves the index unchanged.
- **Memory only:** saving and loading indexes is not implemented yet.

## Documentation

| Document | What's inside |
|---|---|
| [Getting started](docs/getting-started.md) | Building on every platform, build options, using hnsw-lite in your project, troubleshooting |
| [User guide](docs/user-guide.md) | Everything the library does and how to use it: indexes, metrics, metadata, filters, batch and range search, threads, errors |
| [API reference](docs/api-reference.md) | Every public class and function |
| [Performance and tuning](docs/performance.md) | Benchmarks and how to choose settings |
| [Architecture and design](docs/architecture.md) | How it works inside: memory layout, SIMD kernels, the HNSW graph, filters and the query planner |
| [Testing](docs/testing.md) | The 418 tests, code coverage, continuous integration, and bugs found by testing |
| [Project notes](docs/project-notes.md) | Limitations, roadmap, comparison with hnswlib, Faiss, Qdrant and Milvus, references |

## Project status

Done: Flat and HNSW indexes, SIMD kernels, real deletion, metadata, filtered search, batch search and range search. Next: updating vectors in place, saving and loading, and safe concurrent inserts. See the [roadmap](docs/project-notes.md#roadmap).

## License

Released under the [MIT License](LICENSE).