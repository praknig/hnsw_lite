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
| **Collections** | `Collection`: documents with any of a dense vector, text (BM25) and a learned sparse vector, plus metadata, under one ID space; hybrid search by default; durable as a whole; automatic maintenance (upkeep, rebuilds, checkpoints) on a background thread or on request; presets (balanced, high recall, low memory); opt-in document expiry (TTL) and string keys (any string, such as names or UUIDs) |
| **Index types** | `HnswIndex` (graph, approximate); `FlatIndex` (exact), optionally an **inverted file (IVF, IVF-PQ)**; `DiskIndex` (DiskANN-style Vamana graph on disk, read-only, built within a memory budget); `SparseIndex` (inverted index for sparse vectors, exact) |
| **Metrics and speed** | Squared L2, inner product (graphs kept connected when vector lengths vary widely), cosine; SIMD kernels (AVX2, AVX-512, NEON, scalar) chosen at run time, with identical results on every CPU; opt-in graph reordering for memory locality |
| **Search** | k-nearest search with per-query beam width; batch search on a thread pool; range search; search by stored ID; search exactly among given IDs (`SearchOptions::among`); deadlines, cancellation and progress reports; search statistics; opt-in re-ranking by metadata (recency decay, confidence multipliers, boosts) and MMR diversity |
| **Collection search** | Group-by (`search_groups`: the best documents rather than many chunks of one, each group filled from its own members); recommendations from positive and negative example documents (`recommend`); faceted search (top values, numeric range buckets, distinct counts, over everything, a filter's matches or the top hits) |
| **Hybrid and text search** | Sparse vectors (SPLADE, or BM25 from the built-in `Bm25` encoder); `hybrid_search()` fusing dense and sparse results by Reciprocal Rank Fusion or weighted scores; opt-in text analysis for BM25: the Snowball English stemmer, stop words, Unicode-aware tokens |
| **Metadata and filters** | Typed fields (integer, float, boolean, keyword, tags), strict or dynamic schemas; filters with comparisons, ranges, membership, tags, geographic radius and box (across the antimeridian) and boolean logic, plus custom predicates; a query planner choosing exact search, ACORN filtered graph search or plain graph search by selectivity; payload indexes with exact counts and posting lists |
| **Records and changes** | Add, `add_batch` (parallel graph building), update, upsert, real deletion with graph repair and slot reuse, `remove_batch`, bulk changes by filter (`remove_where`, `set_metadata_where`, `count_where`); reading back (`get_vector`, `get`, paged `scan`) |
| **Compression** | SQ8, binary, product quantization (PQ) and RaBitQ, each with optional re-ranking at full precision and SIMD code kernels; Matryoshka truncation; keeping only codes to save memory; half precision (float16, bfloat16): vectors halved, recall within 0.005, also in the disk index |
| **Storage** | Versioned, checksummed file format (older versions still load); atomic saves; deterministic files, identical across platforms |
| **Durability** | `open_durable()`: a write-ahead log plus snapshots, crash recovery to the last acknowledged change, three flush modes (`Buffered`, `Grouped`, `Synced`), `checkpoint()`, one writer per directory |
| **Health and maintenance** | `health()` on every index (reachability, degrees, estimated recall, codec and IVF drift, churn); incremental `upkeep()` with budgets; `compact()` in parallel with the same result on any thread count, and automatically; `rebuild()` while searches and changes continue, retraining codecs and lists |
| **Concurrency** | Every index thread-safe; searches run in parallel; changes run one at a time under one lock per index, and writers are never starved. On an `HnswIndex`, an insertion finds its neighbors alongside searches, which wait only while it links; a collection's searches wait for a whole change |
| **Tools and other languages** | The `hnsw-lite` command-line tool (`inspect`, `verify`, `health`, `export` to JSON Lines or CSV); Python bindings with NumPy and pandas, and wheels for Linux, macOS and Windows; an hnswlib-compatible Python module; Arrow and Parquet in and out; importers from hnswlib files, FAISS indexes and Chroma collections; reranking hooks (cross-encoders); benchmark tools (a comparison with hnswlib, FAISS and USearch, a stress test, a nightly job) |
| **Robustness** | Invalid input rejected with clear exceptions; every change all-or-nothing, including under out-of-memory; 961 tests, sanitizers, coverage gates, fuzzing of every file and log loader, emulated older CPUs, real process kills |

## Compared with similar libraries

Summarized from each project's public documentation as of October 2026; these projects move quickly, so check their current documentation before deciding.

### Embedded libraries

Libraries linked into your program, like hnsw-lite.

| | Index types | Compression | Filtering | Updates and deletes | Crash-safe log | Sparse and hybrid | Languages |
|---|---|---|---|---|---|---|---|
| **hnsw-lite** | HNSW, flat, IVF, disk graph (Vamana), sparse | SQ8, binary, PQ, RaBitQ, half precision, Matryoshka | Typed metadata, geo filters, query planner, ACORN, payload indexes, facets | Real deletion with graph repair; upkeep and rebuilds | **Yes** | **Yes** (BM25 with stemming, SPLADE, RRF) | C++, Python |
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

**Gaps.** Reach: C++ and Python only, with no C API, no packages for vcpkg or Conan, and no other language bindings yet. Scale: no GPU support and no memory-mapped loading; very long posting lists are scanned without pruning, which limits text and hybrid search on large corpora. Retrieval: no multivector (ColBERT-style) search yet, and text analysis is English-first. Benchmarks: results on SIFT1M and GloVe against hnswlib, FAISS and USearch are published on the `benchmarks` branch, on one cloud machine class; other hardware is still to be measured. The open features below address these.

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
| 7 | **Command-line tool:** inspect, verify, report health, compact, convert and export | **done**: `inspect`, `verify`, `health`, `export`; open: `compact` and `convert` | 1 day |
| 8 | **Import and export:** NumPy, `.fvecs` and `.ivecs`, Parquet, metadata as JSON | **done**: NumPy, Arrow and Parquet, importers from hnswlib, FAISS and Chroma, export to JSON Lines or CSV; open: loading `.fvecs` and `.ivecs` into an index (the benchmarks read them) | 1 day |
| 9 | **Observability:** a metrics callback (latency, log size, checkpoints) and a logging hook | | 2 days |
| 10 | **Published benchmarks:** SIFT1M, GloVe and a modern embedding set, against hnswlib, USearch and FAISS | **done**: `bench/python/compare.py`, and results on SIFT1M and GloVe published on the `benchmarks` branch (`real-datasets/`) | |
| 11 | **RaBitQ quantization** | **done** (`CodecKind::RaBitQ`); open: re-ranking pools sized for it (one-bit codes need far more candidates) | 2 days |
| 12 | **Half-precision storage** (f16, bf16) | **done**, also in the disk index; open: float16 originals for cosine indexes | 2 days |
| 13 | **Memory-mapped loading** of saved indexes | | 3 days |
| 14 | **Multivector records:** several vectors per record (named, or chunks), and late-interaction (ColBERT) scoring | | 1 week |
| 15 | **Faster sparse search** for long posting lists (~~a write-ahead log for a standalone `SparseIndex`~~: **done**) | MaxScore was tried and searched 4× slower; next: block-max WAND | 1–2 weeks |
| 16 | **Filtering:** ~~ACORN filtered graph search~~, ~~date fields~~, ~~payload posting lists~~, ~~facet counts~~, ~~geographic fields~~ (all **done**) | Open: traversal aware of filters correlated with the data (matches clustered away from the query) | 1–2 weeks |
| 17 | **Search extras:** ~~diversity re-ranking (MMR)~~, ~~recommendations from positive and negative examples~~ (**done**); automatic beam width for a target recall, Hamming and L1 metrics | | 1 week |
| 18 | **Records:** ~~string and UUID IDs~~; ~~expiry (TTL)~~ (**done**, for collections) | | |
| 19 | **Concurrency:** searches that do not wait for changes | **Partly done**: an `HnswIndex` insertion no longer holds searches up while it searches; open: `add_batch` and collection changes, and write throughput under heavy reading (finer-grained locks) | 1–1.5 weeks |
| 20 | **Quality:** ~~fuzzing the file and log loaders~~ (**done**); splitting the test program by group for faster builds; Windows on ARM | | 3 days |
| 21 | **Optional HTTP server** around the library | | 3–4 days |
| 22 | **Memory layout and prefetching** for indexes larger than the cache (vectors next to their links, neighbors fetched ahead) | Measured: single-thread insert cost doubles from 100,000 to 2,000,000 vectors, a cache effect | 1–2 weeks |

## Left out deliberately

**Sharding, replication, encryption at rest and access control.** These belong to a database server such as Qdrant or Milvus, not to a library. The optional HTTP server (21) is as far as hnsw-lite should go in that direction.

## Suggested order

| Stage | Features | Why |
|---|---|---|
| **A. One object, no babysitting** | ~~1, 2, 3~~ (done) | Turns the toolkit into a product: one API, maintained automatically |
| **B. Reach** | ~~4~~ (done), 5, 6 | Python first, then C++ packaging and other languages |
| **C. Operating it** | ~~7, 8~~ (mostly done), 9 | Looking inside indexes, moving data in and out, watching them in production |
| **D. Confidence** | ~~10~~ (done), 20 | Published numbers on standard datasets (done); faster builds |
| **E. Speed at scale** | 19, 22, 15 | Writers and readers both fast; memory locality; text search on large corpora |
| **F. Quality of results** | 11, 12, 13, 14, 16, 17 | As real usage shows which matter most |

Done so far: stage A (collections, their automatic maintenance, presets); the core (deletion, updates, saving, thread safety, parallel building); records and bulk changes; search by ID, among given IDs, deadlines and cancellation; compression (SQ8, binary, PQ, RaBitQ, half precision, Matryoshka truncation, codes only); IVF and the disk-based index; health reports, upkeep and rebuilds; durability; hybrid search with BM25 text analysis; filtering with geo filters, payload posting lists and faceted search; group-by and recommendations; string keys and expiry; the command-line tool; Python bindings, wheels, an hnswlib-compatible module, Arrow and Parquet, and importers; benchmark tools. Most recently: connected inner-product graphs, group-by filled from group members, insertions that leave searches running, a parallel deterministic `compact()`, and a leaner search loop.