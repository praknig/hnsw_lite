# hnsw-lite

[![CI](https://github.com/praknig/hnsw_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/praknig/hnsw_lite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, fast, in-memory **vector search library** in modern C++20.

Store vectors (for example, text or image embeddings) under your own IDs, optionally with metadata, then find the vectors closest to a query: **exactly** with a Flat index, or **approximately and much faster** with an HNSW graph. Searches can be filtered by metadata, run in batches, or return everything within a distance.

hnsw-lite was built from scratch to show how libraries like hnswlib, Faiss and Qdrant work inside, so the code is meant to be read as well as used. It is tested thoroughly: 894 tests, 100% line coverage, memory and thread checkers, and CI on Linux, Windows and macOS.

## Features

- **Four index types:** `HnswIndex` (approximate graph search: about 14x faster than exact at 99.8% recall on 50,000 vectors), `FlatIndex` (exact, optionally an inverted file), `DiskIndex` (a DiskANN-style graph on disk, for indexes larger than memory) and `SparseIndex` (sparse vectors and BM25, for hybrid search).
- **Three distance metrics:** squared Euclidean (L2), inner product and cosine.
- **SIMD speed on any CPU:** AVX2, AVX-512 or ARM NEON versions are chosen automatically at runtime.
- **Updates and real deletion:** `update` and `upsert` replace a vector without ever losing the old one on failure; removed IDs can be reused immediately, and `compact()` rebuilds a heavily changed index.
- **Metadata and filtered search:** integer, float, boolean, keyword and tag fields; filters like `Filter::eq("topic", "tech") && Filter::ge("year", 2020)`; a query planner that picks the fastest strategy.
- **Batch search** on a thread pool and **range search** (everything within a radius).
- **Parallel building:** `add_batch` links many vectors into the HNSW graph on several threads, all or nothing.
- **Bulk changes by filter:** `remove_where`, `set_metadata_where` and `count_where`, like SQL's DELETE and UPDATE with a WHERE clause.
- **Reading records back:** `get_vector` (exactly as given), `get`, and `scan` to page through records by ID, optionally filtered.
- **Opt-in result controls:** recency decay, confidence multipliers and boosts by metadata; MMR diversity; document expiry (TTL); string and UUID keys; dates.
- **Python bindings:** `pip install .`, with NumPy arrays in and out, pandas DataFrames for loading and results, and searches that release the GIL.
- **Collections:** one object for documents with any of a dense vector, text (BM25) and a learned sparse vector, plus metadata, searched together, durable as a whole, and maintained automatically (upkeep, rebuilds, checkpoints); presets choose the settings.
- **Hybrid search:** sparse vectors (BM25, SPLADE) in an exact inverted index, fused with dense results by Reciprocal Rank Fusion.
- **Durability:** a write-ahead log with snapshots: `open_durable()` recovers every acknowledged change after a crash, with three flush modes.
- **Health reports and upkeep:** reachability, degrees, estimated recall and drift on every index type; incremental `upkeep()` (like a database's vacuum), `rebuild()` retraining compression and IVF lists while the index stays in use, and optional automatic compaction.
- **Disk-based index (DiskANN-style):** indexes larger than memory, built within a memory budget and searched with a few disk reads per query (concurrently), with filters.
- **Inverted file (IVF, and IVF-PQ):** scan only the lists nearest to a query, with filters and every codec.
- **Compression (SQ8, binary, PQ, half precision) and Matryoshka truncation:** one byte or one bit per dimension, one byte per part of the vector (PQ), or two bytes per dimension (float16 and bfloat16, which halve vector memory with almost no recall lost), searched with codes and re-ranked at full precision; or index only the first dimensions of embeddings trained for it; optionally keep only the codes (about a quarter of the memory) (more codecs and index types are [planned](docs/compression-and-index-types-plan.md)).
- **Search by stored ID, deadlines and cancellation:** "more like record 42"; searches that stop on time with the best results so far; long operations with progress reports that can be cancelled and are then undone.
- **Save and load:** one call writes the whole index to a file and one reads it back, checked against damage and portable between platforms.
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

## Python

```bash
pip install hnsw-lite-lib   # or "pip install ." from the repository; import hnsw_lite_lib
```

```python
import numpy as np
import hnsw_lite_lib as hl

with hl.Collection(dim=384, path="docs.db") as docs:                 # durable; omit path for in memory
    docs.upsert(1, vector=embedding, text="The quick brown fox", metadata={"lang": "en"})
    hits = docs.search(vector=query, text="fox", k=10, filter={"lang": "en"})
    docs.upsert_dataframe(df, id_column="id", vector_column="embedding", text_column="body")   # pandas
    table = docs.search(text="fox", k=10, as_dataframe=True)

index = hl.HnswIndex(dim=128, metric=hl.Metric.L2)
index.add(np.arange(10_000), vectors)
ids, distances = index.search_batch(queries, k=10)                  # NumPy arrays of shape (n, 10)
```

NumPy arrays of any float type go in and come out; searches release the GIL, so Python threads search in parallel. See the [Python guide](docs/python.md).

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

### Update, remove and compact

```cpp
index.update(101, new_vector);   // replace a vector (metadata is kept)
index.upsert(200, new_vector);   // add it if new, replace it otherwise
index.remove(100);               // the ID can be added again immediately
index.compact();                 // after many changes: rebuild for the best quality
```

### Save and load

```cpp
index.save("my.index");                          // everything: vectors, graph, metadata, settings
auto loaded = vecdb::HnswIndex::load("my.index");  // std::unique_ptr<HnswIndex>, ready to search
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
- **Thread safety:** every index can be used from any number of threads with no lock of your own. Searches and saves run in parallel; changes run one at a time and wait for running searches. Changes cannot be made from inside a search (for example from a filter predicate): that throws `std::logic_error`.
- **Errors:** invalid input (wrong dimension, NaN, a duplicate ID, a metadata type mismatch) throws `std::invalid_argument` and leaves the index unchanged.
- **Saving:** `index.save(path)` writes everything, and `HnswIndex::load(path)` returns the index exactly as it was; a damaged file throws `vecdb::FormatError`.

## Nightly benchmark

A workflow runs the stress and load test every night at the largest size the machine's memory allows (up to 50 million vectors) and publishes the results to the [`benchmarks` branch](https://github.com/praknig/hnsw_lite/tree/benchmarks). These charts update from there automatically:

![Nightly benchmark trends](https://raw.githubusercontent.com/praknig/hnsw_lite/benchmarks/history.png)

![Latest recall and speed](https://raw.githubusercontent.com/praknig/hnsw_lite/benchmarks/latest-recall.png)

See [Benchmarks](docs/benchmarks.md#nightly-benchmark) for running it yourself.

## Documentation

| Document | What's inside |
|---|---|
| [Getting started](docs/getting-started.md) | Building on every platform, build options, using hnsw-lite in your project, troubleshooting |
| [User guide](docs/user-guide.md) | Everything the library does and how to use it: indexes, metrics, metadata, filters, batch and range search, threads, errors |
| [Developer reference](docs/developer-reference.md) | Every method and parameter in detail: what it does, what it expects, what can go wrong, and its limits |
| [Python](docs/python.md) | The Python package: installing, collections, indexes, filters, NumPy and pandas, threads and errors |
| [Benchmarks](docs/benchmarks.md) | Comparing hnsw-lite with hnswlib, FAISS and USearch on your machine: running the script, fairness, reading results |
| [API reference](docs/api-reference.md) | Every public class and function |
| [Performance and tuning](docs/performance.md) | Benchmarks and how to choose settings |
| [Architecture and design](docs/architecture.md) | How it works inside: memory layout, SIMD kernels, the HNSW graph, filters and the query planner |
| [Testing](docs/testing.md) | The 894 tests, code coverage, continuous integration, and bugs found by testing |
| [Project notes](docs/project-notes.md) | Limitations, the roadmap, and references |
| [Core capabilities plan](docs/core-capabilities-plan.md) and [its test plan](docs/core-capabilities-test-plan.md) | The original designs and test scenarios for deletion, updates, saving and loading, and thread safety (all implemented) |
| [Hardening and performance plan](docs/hardening-and-performance-plan.md) | Edge cases, optimizations and their order, with the status of each phase |
| [Features](docs/feature-catalog.md) | Everything hnsw-lite has, a comparison with similar libraries and databases, and what could come next, in a suggested order |
| [Compression and index types plan](docs/compression-and-index-types-plan.md) | Quantization (SQ8, binary, PQ), Matryoshka truncation, an IVF index and a disk-based index: phases, decisions and testing |

## Project status

Done: Flat and HNSW indexes, SIMD kernels, updates and real deletion, metadata, filtered search, batch search, range search, saving and loading (file format version 3), thread-safe indexes and parallel building (`add_batch`), bulk changes by filter, reading records back, search by stored ID, deadlines and cancellation, and compression (SQ8, binary and PQ codes with SIMD kernels, Matryoshka truncation, keeping only codes), an inverted file (IVF, IVF-PQ), and a disk-based index. Next: see the [roadmap](docs/project-notes.md#roadmap).

## License

Released under the [MIT License](LICENSE).