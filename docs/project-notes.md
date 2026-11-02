# Project notes

Why hnsw-lite exists, its current limitations, the roadmap, how it compares with established libraries, and references.

## Why this project

Vector search is the backbone of modern AI infrastructure: retrieval-augmented generation (RAG), semantic search and multimodal retrieval all depend on it. Production vector databases get their speed from low-level C++ work: memory layout, cache efficiency and SIMD math.

hnsw-lite implements that core in a small, readable codebase, with every design decision documented.

## Limitations

These are deliberate for the current stage:

- **Changes run one at a time.** Indexes are thread-safe, and every change takes the lock alone to apply itself. An `add`, `update` or `upsert` on an HNSW index finds and chooses its neighbors while searches keep running and holds them up only to link (see the [architecture](architecture.md)); `add_batch` links its vectors on several threads, but searches wait while it runs.
- **Writes slow down under heavy concurrent reading.** Since searches are no longer held up by an insertion's search work, a writer competes with them for the cores: with 10 to 19 readers on 20 threads, upserts and adds ran 2.4 to 8 times slower than with readers kept out, while the readers ran 2 to 8 times faster. Finer-grained locking would serve both.
- **Heavy update churn weakens the graph until upkeep.** Each update links a new version and repairs around the old one; small losses add up. Measured on the current code with test `c14`'s 20 vectors and its 6,400 upserts (320 per vector) applied in 200 random orders: a vector became unreachable in 1 order (at worst 1 of the 20), and none after `upkeep()`, which reconnects them; removal repair keeps each former neighbor's incoming link (see the [user guide](user-guide.md#upkeep)). `compact()` still rebuilds the graph completely. (hnswlib has the same weakness with repeated updates.)
- **Inner product with vectors of very different lengths:** neighbors are chosen by angle, so the graph stays connected and searches find their answers (measured on 500,000 recommendation-like vectors: recall@10 0.993 at `ef` 64; filtered graph searches found every right answer). Vectors of very small length, though, are rarely any other vector's nearest by inner product, so many have no link pointing to them (a health report counts them unreachable); they are found as answers when a filter or `among` asks for them. If lengths carry no meaning in your data, normalize and use cosine.
- **Cosine indexes use twice the memory for vectors**, keeping each vector as given (for `get_vector`) besides the normalized copy used for searching.
- **Graphs made almost entirely of identical vectors keep some of them unreachable until upkeep.** Neighbor selection now keeps copies of a vector (up to the room distinct neighbors do not need) and breaks distance ties by a hash of the pair: with a dozen copies per value nothing becomes unreachable; with 600 identical vectors and M 8 about 10% do (98.5% before), and `upkeep()` reconnects them. Run it after building heavily duplicated data.
- **HNSW build speed.** One-by-one adds are single-threaded (use `add_batch` to build on several threads), and every access is bounds-checked for safety. Unchecked accessors in the hot loops are future optimizations.
- **A plain `save()` does not force data onto the physical disk:** it is atomic (temporary file, then rename), but a power cut right after it returns could lose it. Durable directories (`open_durable()`) flush their log and snapshots to the disk itself.
- **HNSW reclaims memory by reuse.** A removed vector's slot is reused by the next insert, so memory stays bounded by the peak number of live vectors, but it is not returned to the system until `compact()` is called.
- **Index IDs are 64-bit integers.** Collections can also take string keys (`string_ids`): each is hashed to a 64-bit ID, and two keys with the same hash are refused, never mixed up.
- **Text analysis is English-first:** stemming and stop words are English; other languages get Unicode-aware tokens (`unicode`) without stemming. No synonyms.
- **Sparse search is exact without pruning:** very common terms make long posting lists (MaxScore pruning was tried and searched slower; block-max WAND is on the roadmap). A standalone `SparseIndex` can be durable (`open_durable`), as collections are.
- **Group-by with very large groups** (over 4,096 documents per value) fills short groups with the full filtered search rather than a search among the group's members; grouping by a field with few, large values (a category) stays slow.
- **One process per durable directory**, enforced with a lock file; a checkpoint holds changes (not searches) while its snapshot is written.
- **Filters are limited to 256 levels of nesting.** Combine many conditions with `in()`, `has_any_tag()` or a balanced tree instead.
- **Dynamic metadata fields are never removed**, even when no vector uses them: field names are the schema, and removing one automatically would silently forget its type. Unused strings, by contrast, are reclaimed.
- **Filter-aware graph construction is not implemented.** Very selective filters go to exact search; between roughly 1% and 10% selectivity the planner's thresholds matter most, and may need tuning per dataset.
- **HNSW range search is approximate**, like k-nearest search; use `Strategy::ForceExact` when every vector in the radius must be found.
- **Custom predicates must be thread-safe** when used with batch search, and the planner can only estimate them by sampling.
- **Inner product costs more per search than L2 and cosine.** It is not a true distance: on data with very different lengths, a healthy graph needs more distance computations per query (measured on 500,000 recommendation-like vectors at `ef` 64: 1.35 times as many, and half the queries per second), and building took 2.3 times as long.
- **Very small `M` builds a sparse graph.** `M = 2` works but leaves about 10% of nodes unreachable in tests; use `M >= 8`.
- **Kernels compare one pair of vectors at a time.** Batched query-vs-many kernels and matrix-multiplication-based flat search are possible future optimizations.
- **On some older Intel CPUs, heavy AVX-512 use lowers the clock speed.** If the AVX2 version benchmarks faster on your machine, that is why.

## Roadmap

Every feature that could come next, consolidated and in a suggested order, is in the [feature catalog](feature-catalog.md). Compression and new index types (quantization, Matryoshka truncation, IVF and a disk-based index) have their own [plan](compression-and-index-types-plan.md).

- [x] **Layer 1:** aligned vector store, arena allocator, graph storage, ID mapping, tests
- [x] **Layer 2:** SIMD distance kernels (scalar, AVX2, AVX-512, NEON), runtime dispatch, tests, benchmark
- [x] **Layer 3a:** Flat index with top-k heap, used as ground truth
- [x] **Layer 3b:** HNSW index: random levels, insertion, neighbor heuristic, beam search, visited lists
- [x] Search benchmark on synthetic data (recall@10 vs. queries per second)
- [x] Continuous integration: Linux (GCC, Clang, sanitizers), Windows (MSVC, MinGW) and macOS on ARM
- [x] Real deletion: reusable IDs, slot reuse, graph repair, `compact()`
- [x] Updating vectors: `update` and `upsert`
- [x] Saving and loading indexes
- [x] Thread-safe indexes: one lock per index (stage A)
- [x] Parallel building: `add_batch` links vectors on several threads (stage B)
- [x] Bulk changes by filter: `remove_where`, `set_metadata_where`, `count_where`
- [x] Reading records back: `get_vector`, `get`, `scan` (file format version 2)
- [x] `remove_batch`, `search_by_id`, and deadlines, cancellation and progress
- [x] Compression: k-means and the codec interface; SQ8 and binary codes with re-ranking; Matryoshka truncation; keeping only codes (file format version 3)
- [x] SIMD kernels for code distances (SQ8, binary)
- [x] Product quantization
- [x] An inverted file (IVF, and IVF-PQ) for `FlatIndex`
- [x] A disk-based index, read-only (see the [plan](compression-and-index-types-plan.md))
- [x] Health reports for every index type (step 1 of index upkeep)
- [x] Index upkeep: better repair under churn, incremental `upkeep()`, automatic `compact()` (step 2)
- [x] Rebuilds while in use, with an atomic swap, retraining codecs and IVF lists (step 3)
- [x] Durability: a write-ahead log with snapshots and crash recovery (`open_durable()`)
- [x] Sparse vectors, BM25 and hybrid search with rank fusion (`SparseIndex`, `hybrid_search()`)
- [x] Write-ahead log for `SparseIndex`
- [ ] MaxScore pruning for long posting lists. Tried: an exact document-at-a-time MaxScore read 29% fewer postings but searched 4× slower than the term-at-a-time loop (measured on 50,000 SPLADE-like documents, k = 10 and 100): finding the next document across the query's cursors costs more than the postings it saves. A faster version needs block-max indexes or skip structures.
- [x] Fuzzing the file and log loaders
- [x] Collections: documents over dense, text and sparse sides, durable as a whole (`Collection`)
- [x] Automatic maintenance of collections (`MaintenancePolicy`, `maintain()`, a background thread)
- [x] Presets for collections (`CollectionOptions::preset`)
- [x] ACORN filtered graph search, chosen by the query planner
- [x] RaBitQ compression (one bit per dimension and two numbers; helps most for L2 and wide vectors)
- [x] Python bindings with NumPy and pandas integration (`hnsw_lite_lib`)
- [x] Python wheels for Linux, macOS and Windows (`pip install hnsw-lite-lib`)
- [x] C++ packaging: an installable CMake package (`find_package(hnsw_lite)`), a vcpkg port and a Conan recipe
- [ ] A C API (see the [suggested order](feature-catalog.md#suggested-order))
- [x] Search features: metadata, filtered search with a query planner, payload index (counts), batch search, range search
- [x] Payload index posting lists; faceted search (top values, numeric ranges, distinct counts)
- [ ] Filter-aware graph construction. Measured first (30,000 vectors, filters from 0.5% to 30%, correlated with the clusters or not, indexed or ranges): recall was already 0.995 or better everywhere; the gap was cost, for correlated filters past exact search's limits. Built instead: the planner's exact fallback (4.3 to 6.2× faster there). Construction changes stay open for data where recall does fall.
- [x] Graph reordering for memory locality (`reorder()`), score modifiers (recency decay, confidence, boosts), dates, MMR diversity, document expiry (TTL), string keys
- [x] Benchmark tools: a comparison with hnswlib, FAISS and USearch, a stress test at a million vectors and more, a nightly job publishing charts
- [x] Half-precision vectors (float16, bfloat16), a compact ID map, and collection settings (expiry, string keys) saved with the collection
- [x] Benchmark results on SIFT1M and GloVe, published on the `benchmarks` branch (`real-datasets/`, from 2026-10-28)
- [x] Collections: scanning every document, and exporting to a DataFrame (`scan()`, `documents()`, `to_pandas()`)
- [x] Faceted search: counts by field over the top hits and the whole corpus, filtered counts, numeric ranges
- [x] A command-line tool: `inspect`, `verify`, `health` and `export` for any file
- [ ] A first release on PyPI (v0.1.0), with a changelog
- [ ] LangChain and LlamaIndex integrations; an ann-benchmarks adapter
- [x] Integer SIMD kernels for SQ8 distances (inner product and cosine)
- [x] Half precision for the disk index (float16, bfloat16)
- [x] Group-by (`search_groups`): the best documents rather than many chunks of one
- [x] Recommendations from example documents (`recommend`)
- [x] Geo filters (`geo_radius`, `geo_box`) over two numeric fields, across the antimeridian
- [x] Text analysis for BM25: the Snowball English stemmer, stop words, Unicode-aware tokens (opt-in, saved with a collection)
- [x] Python: reranking hooks (cross-encoders), Arrow and Parquet import and export, importers from hnswlib files, FAISS and Chroma, an hnswlib-compatible module
- [x] A profiling review: quadratic bulk removals, insertions and filter compilation made linear or O(n log n); filters evaluated a column at a time; lighter hybrid fusion; a faster checksum (see [Performance](performance.md#profiling-review-complexity-and-hot-paths))
- [x] Inner-product graphs that stay connected when lengths vary widely (neighbors chosen by angle, lists filled up)
- [x] Group-by filling short groups from their members; searching among given IDs (`SearchOptions::among`)
- [x] Insertions decided under the shared lock: `add`, `update` and `upsert` (and an update's repair) let searches run meanwhile
- [x] Parallel, deterministic `compact()` (`OperationOptions::threads`; the same index whatever the thread count)
- [x] A leaner search loop: 8.7% fewer instructions, 15 to 19% faster measured, the same results
- [ ] Write throughput under heavy concurrent reading: finer-grained locking (per node, as hnswlib does) or priority for a writer's final step
- [ ] Collections: let searches run during a document's change, as an `HnswIndex` does for its insertions (a collection holds its own lock for the whole change)
- [ ] Memory layout and prefetching for indexes larger than the cache (vectors next to their links; neighbors fetched ahead)
- [ ] Block-max WAND for long posting lists (BM25 text and hybrid search at scale)
- [ ] Filter-aware traversal for filters correlated with the data (ACORN for clustered matches)
- [ ] Re-ranking pools sized per codec (RaBitQ and binary need far more candidates), and float16 originals for cosine indexes
- [ ] Multi-vector documents (several vectors per document, late interaction)
- [ ] Read-only, memory-mapped opening shared by several processes

The core capabilities (real deletion, updating vectors, saving and loading, thread safety and parallel building, all done) were planned in [docs/core-capabilities-plan.md](core-capabilities-plan.md), with 119 test scenarios in [docs/core-capabilities-test-plan.md](core-capabilities-test-plan.md). The [hardening and performance plan](hardening-and-performance-plan.md) schedules the remaining work together with edge-case fixes and optimizations.

## Comparison with existing implementations

The comparison with hnswlib, USearch, FAISS, Annoy, ScaNN, Voyager, DiskANN, sqlite-vec and the vector databases LanceDB, Chroma, Qdrant and Milvus is in the [feature catalog](feature-catalog.md#compared-with-similar-libraries).

## References

- Yu. A. Malkov and D. A. Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv:1603.09320.
- [hnswlib](https://github.com/nmslib/hnswlib)
- [Faiss](https://github.com/facebookresearch/faiss)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Milvus](https://github.com/milvus-io/milvus)
- [USearch](https://github.com/unum-cloud/usearch)
- S. Jayaram Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019.
- L. Patel et al., *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*, SIGMOD 2024.
- J. Gao and C. Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*, SIGMOD 2024.
- G. V. Cormack, C. L. A. Clarke and S. Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009.
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)
