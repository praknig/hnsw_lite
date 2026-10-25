# Features

Everything hnsw-lite does today, how it compares with similar libraries, and what could come next.

## Contents

- [What hnsw-lite has](#what-hnsw-lite-has)
- [Compared with similar libraries](#compared-with-similar-libraries)
- [Open features](#open-features)
- [Left out deliberately](#left-out-deliberately)
- [Suggested order](#suggested-order)

## What hnsw-lite has

A C++20 library with no dependencies, tested on Linux, Windows and macOS (x86 and ARM). The [user guide](user-guide.md) explains each feature; the [API reference](api-reference.md) lists every call.

| Area | Features |
|---|---|
| **Collections** | `Collection`: documents with any of a dense vector, text (BM25) and a learned sparse vector, plus metadata, under one ID space; hybrid search by default; durable as a whole; automatic maintenance (upkeep, rebuilds, checkpoints) on a background thread or on request; presets (balanced, high recall, low memory); opt-in document expiry (TTL) and string or UUID keys |
| **Index types** | `HnswIndex` (graph, approximate); `FlatIndex` (exact), optionally an **inverted file (IVF, IVF-PQ)**; `DiskIndex` (DiskANN-style Vamana graph on disk, read-only, built within a memory budget); `SparseIndex` (inverted index for sparse vectors, exact) |
| **Metrics and speed** | Squared L2, inner product, cosine; SIMD kernels (AVX2, AVX-512, NEON, scalar) chosen at run time, with identical results on every CPU; opt-in graph reordering for memory locality |
| **Search** | k-nearest search with per-query beam width; batch search on a thread pool; range search; search by stored ID; deadlines, cancellation and progress reports; search statistics; opt-in re-ranking by metadata (recency decay, confidence multipliers, boosts) and MMR diversity |
| **Hybrid search** | Sparse vectors (SPLADE, or BM25 from the built-in `Bm25` encoder); `hybrid_search()` fusing dense and sparse results by Reciprocal Rank Fusion or weighted scores |
| **Metadata and filters** | Typed fields (integer, float, boolean, keyword, tags), strict or dynamic schemas; filters with comparisons, ranges, membership, tags and boolean logic, plus custom predicates; a query planner choosing exact search, ACORN filtered graph search or plain graph search by selectivity; payload indexes with exact counts |
| **Records and changes** | Add, `add_batch` (parallel graph building), update, upsert, real deletion with graph repair and slot reuse, `remove_batch`, bulk changes by filter (`remove_where`, `set_metadata_where`, `count_where`); reading back (`get_vector`, `get`, paged `scan`) |
| **Compression** | SQ8, binary and product quantization (PQ), each with optional re-ranking at full precision and SIMD code kernels; Matryoshka truncation; keeping only codes to save memory; half precision (float16, bfloat16): vectors halved, recall within 0.005 |
| **Storage** | Versioned, checksummed file format (older versions still load); atomic saves; deterministic files, identical across platforms |
| **Durability** | `open_durable()`: a write-ahead log plus snapshots, crash recovery to the last acknowledged change, three flush modes (`Buffered`, `Grouped`, `Synced`), `checkpoint()`, one writer per directory |
| **Health and maintenance** | `health()` on every index (reachability, degrees, estimated recall, codec and IVF drift, churn); incremental `upkeep()` with budgets; automatic `compact()`; `rebuild()` while searches and changes continue, retraining codecs and lists |
| **Concurrency** | Every index thread-safe; searches run in parallel; changes serialized under one lock per index; writers never starved |
| **Robustness** | Invalid input rejected with clear exceptions; every change all-or-nothing, including under out-of-memory; 903 tests, sanitizers, coverage gates, emulated older CPUs, real process kills |

## Compared with similar libraries

Summarized from each project's public documentation as of October 2026; these projects move quickly, so check their current documentation before deciding.

### Embedded libraries

Libraries linked into your program, like hnsw-lite.

| | Index types | Compression | Filtering | Updates and deletes | Crash-safe log | Sparse and hybrid | Languages |
|---|---|---|---|---|---|---|---|
| **hnsw-lite** | HNSW, flat, IVF, disk graph (Vamana), sparse | SQ8, binary, PQ, Matryoshka | Typed metadata, query planner, ACORN, payload indexes | Real deletion with graph repair; upkeep and rebuilds | **Yes** | **Yes** (BM25, SPLADE, RRF) | C++, Python |
| [hnswlib](https://github.com/nmslib/hnswlib) | HNSW | None | Filter callback | Mark deleted; replace deleted slots | No | No | C++, Python |
| [USearch](https://github.com/unum-cloud/usearch) | HNSW | f16, bf16, i8, binary | Filter callback | Yes | No | No | C++ and 10+ bindings |
| [FAISS](https://github.com/facebookresearch/faiss) | Flat, IVF, HNSW, NSG and many more | SQ, PQ, OPQ, RaBitQ, and more | ID selectors | Depends on the index type | No | No | C++, Python; GPU |
| [Annoy](https://github.com/spotify/annoy) | Random projection forest | None | No | Build once, read only | No | No | C++, Python |
| [ScaNN](https://github.com/google-research/google-research/tree/master/scann) | Partitioning plus anisotropic quantization | Anisotropic quantization | Limited | Mostly built once | No | No | Python (C++ core) |
| [Voyager](https://github.com/spotify/voyager) | HNSW (based on hnswlib) | float8, e4m3 | No | Mark deleted | No | No | Python, Java |
| [DiskANN](https://github.com/microsoft/DiskANN) | Vamana in memory and on SSD | PQ | Label filters | Yes (in-memory index) | No | No | C++, Python, Rust |
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | Exact scan | int8, binary | SQL | Yes (SQL) | Yes, through SQLite | Through SQLite full-text search | SQL in any language |

### Vector databases

Systems with their own storage engine or server, for context: hnsw-lite gives up their servers and clustering, and keeps much of their engine.

| | Kind | Index types | Durability | Hybrid search | Distribution |
|---|---|---|---|---|---|
| [LanceDB](https://github.com/lancedb/lancedb) | Embedded database | IVF-PQ, IVF with HNSW, RaBitQ | Versioned columnar files | Full-text (BM25), hybrid, multivector | No (embedded) |
| [Chroma](https://github.com/chroma-core/chroma) | Embedded or server | HNSW | Yes | Full-text matching | Server version |
| [Qdrant](https://github.com/qdrant/qdrant) | Server | HNSW with payload indexes; scalar, binary, PQ | Write-ahead log | Sparse, RRF and DBSF fusion, multivector | Yes |
| [Milvus](https://github.com/milvus-io/milvus) | Distributed server | HNSW, IVF, DiskANN, GPU indexes | Write-ahead log | Sparse, BM25 full-text, hybrid | Yes |

### Where hnsw-lite stands

**Strengths.** Among embedded libraries, hnsw-lite is the one that behaves like a database engine: crash-safe durability, real deletion that keeps the graph healthy, health reports, incremental upkeep and rebuilds while in use, typed metadata with a query planner, and hybrid search, in one library with no dependencies. It is also unusually well verified: byte-identical results on every platform, all-or-nothing changes under allocation failure, and crash tests that kill real processes.

**Gaps.** Reach: Python and C++ only, with no other language bindings yet. Compression: RaBitQ and half-precision storage, now common elsewhere, are missing. Scale: no GPU support and no memory-mapped loading. Retrieval: no multivector (ColBERT-style) search yet. And there are no published benchmarks against these libraries yet. The suggested order below addresses these.

## Open features

**Effort** is a rough estimate for one developer, including tests and documentation.

| # | Feature | Notes | Effort |
|---|---|---|---|
| 1 | **`Collection` API:** one object for dense vectors, text or sparse vectors, and metadata, under one ID space; hybrid search by default; durable sparse side | **done** | |
| 2 | **Automatic maintenance:** thresholds deciding when to run `upkeep()`, `rebuild()` and `checkpoint()`, from a background thread or a `maintain()` call | **done** for collections (`MaintenancePolicy`) | |
| 3 | **Presets:** balanced, low memory and high recall settings chosen from dimension and size | **done** (`CollectionOptions::preset`) | |
| 4 | **Python bindings,** with wheels for every platform | **done**: `pip install hnsw-lite-lib` (collections, indexes, filters, NumPy, pandas); wheels for Linux, macOS and Windows built and tested by CI, published on version tags | |
| 5 | **Packaging:** vcpkg, Conan, and a CMake package configuration | | 2–3 days |
| 6 | **C API:** a stable interface for Rust, Go, C# and others | | 2 days |
| 7 | **Command-line tool:** inspect, verify, report health, compact, convert and export | | 1–2 days |
| 8 | **Import and export:** NumPy, `.fvecs` and `.ivecs`, Parquet, metadata as JSON | `bench_baseline` already reads `.fvecs` | 1–2 days |
| 9 | **Observability:** a metrics callback (latency, log size, checkpoints) and a logging hook | | 2 days |
| 10 | **Published benchmarks:** SIFT1M, GloVe and a modern embedding set, against hnswlib, USearch and FAISS | | 1 week |
| 11 | **RaBitQ quantization** | Higher recall than binary codes at the same size | 2 weeks |
| 12 | **Half-precision storage** (f16, bf16) | | 2–3 days |
| 13 | **Memory-mapped loading** of saved indexes | | 3 days |
| 14 | **Multivector records:** several vectors per record (named, or chunks), and late-interaction (ColBERT) scoring | | 1 week |
| 15 | **Faster sparse search:** MaxScore pruning for long posting lists; a write-ahead log for a standalone `SparseIndex` | Covered by `Collection` for most users | 3–4 days |
| 16 | **Filtering:** ~~ACORN filtered graph search~~, ~~date fields~~ (**done**: dates as epoch milliseconds, `parse_date`); payload posting lists for exact filtered search, facet counts, geographic fields | | 1–2 weeks |
| 17 | **Search extras:** ~~diversity re-ranking (MMR)~~ (**done**); recommendations from positive and negative examples, automatic beam width for a target recall, Hamming and L1 metrics | | 1 week |
| 18 | **Records:** string and UUID IDs; expiry (TTL) | | 3 days |
| 19 | **Concurrency:** searches that do not wait for long batch changes | | 1–1.5 weeks |
| 20 | **Quality:** fuzzing the file and log loaders; splitting the test program by group for faster builds; Windows on ARM | | 1 week |
| 21 | **Optional HTTP server** around the library | | 3–4 days |

## Left out deliberately

**Sharding, replication, encryption at rest and access control.** These belong to a database server such as Qdrant or Milvus, not to a library. The optional HTTP server (21) is as far as hnsw-lite should go in that direction.

## Suggested order

| Stage | Features | Why |
|---|---|---|
| **A. One object, no babysitting** | ~~1, 2, 3~~ (done) | Turns the toolkit into a product: one API, maintained automatically |
| **B. Reach** | ~~4~~ (done), 5, 6 | Python first, then C++ packaging and other languages |
| **C. Operating it** | 7, 8, 9 | Looking inside indexes, moving data in and out, watching them in production |
| **D. Confidence** | 10, 20 | Published numbers, fuzzing, faster builds |
| **E. Scale and quality of results** | 11–19 | As real usage shows which matter most |

Done so far: stage A (collections, their automatic maintenance, presets), the core (deletion, updates, saving, thread safety, parallel building), records and bulk changes, search by ID and deadlines, compression (SQ8, binary, PQ, Matryoshka), IVF, the disk-based index, health reports, upkeep and rebuilds, durability, and hybrid search.