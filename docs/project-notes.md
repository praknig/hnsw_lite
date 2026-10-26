# Project notes

Why hnsw-lite exists, its current limitations, the roadmap, how it compares with established libraries, and references.

## Why this project

Vector search is the backbone of modern AI infrastructure: retrieval-augmented generation (RAG), semantic search and multimodal retrieval all depend on it. Production vector databases get their speed from low-level C++ work: memory layout, cache efficiency and SIMD math.

hnsw-lite implements that core in a small, readable codebase, with every design decision documented.

## Limitations

These are deliberate for the current stage:

- **Changes run one at a time.** Indexes are thread-safe, and adds, updates and removes take the lock alone. `add_batch` links its vectors on several threads, but searches wait while it runs; letting searches run during inserts would need the lowest storage layer redesigned for concurrent readers.
- **Heavy update churn weakens the graph until upkeep.** Each update links a new version and repairs around the old one; small losses add up. Measured with 320 updates per vector of a 20-vector index: vectors became unreachable in 37 of 200 random orders, 7 since removal repair keeps each former neighbor's incoming link, and none after `upkeep()`, which reconnects them (see the [user guide](user-guide.md#upkeep)). `compact()` still rebuilds the graph completely. (hnswlib has the same weakness with repeated updates.)
- **Inner product with vectors of very different lengths:** vectors whose lengths differ by orders of magnitude leave some vectors hard to reach in an HNSW graph (the long ones dominate every neighborhood), a known property of HNSW with inner product. Normalize the vectors (or use cosine) when their lengths vary widely.
- **Cosine indexes use twice the memory for vectors**, keeping each vector as given (for `get_vector`) besides the normalized copy used for searching.
- **Graphs made almost entirely of identical vectors keep some of them unreachable until upkeep.** Neighbor selection now keeps copies of a vector (up to the room distinct neighbors do not need) and breaks distance ties by a hash of the pair: with a dozen copies per value nothing becomes unreachable; with 600 identical vectors and M 8 about 10% do (98.5% before), and `upkeep()` reconnects them. Run it after building heavily duplicated data.
- **HNSW build speed.** One-by-one adds are single-threaded (use `add_batch` to build on several threads), and every access is bounds-checked for safety. Unchecked accessors in the hot loops are future optimizations.
- **A plain `save()` does not force data onto the physical disk:** it is atomic (temporary file, then rename), but a power cut right after it returns could lose it. Durable directories (`open_durable()`) flush their log and snapshots to the disk itself.
- **HNSW reclaims memory by reuse.** A removed vector's slot is reused by the next insert, so memory stays bounded by the peak number of live vectors, but it is not returned to the system until `compact()` is called.
- **User IDs are 64-bit integers only.**
- **Sparse search is exact without pruning:** very common terms make long posting lists (MaxScore pruning is on the roadmap). A standalone `SparseIndex` can be durable (`open_durable`), as collections are.
- **One process per durable directory**, enforced with a lock file; a checkpoint holds changes (not searches) while its snapshot is written.
- **Filters are limited to 256 levels of nesting.** Combine many conditions with `in()`, `has_any_tag()` or a balanced tree instead.
- **Dynamic metadata fields are never removed**, even when no vector uses them: field names are the schema, and removing one automatically would silently forget its type. Unused strings, by contrast, are reclaimed.
- **Filter-aware graph construction is not implemented.** Very selective filters go to exact search; between roughly 1% and 10% selectivity the planner's thresholds matter most, and may need tuning per dataset.
- **HNSW range search is approximate**, like k-nearest search; use `Strategy::ForceExact` when every vector in the radius must be found.
- **Custom predicates must be thread-safe** when used with batch search, and the planner can only estimate them by sampling.
- **Inner product is harder than L2 and cosine.** It is not a true distance, so recall is lower on un-normalized data (about 0.86 to 0.90 at `ef = 100` in tests, versus 0.997 or higher for L2 and cosine), and a small fraction of short vectors may become unreachable. This matches hnswlib. Use cosine, or normalize vectors, when possible.
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
- [ ] MaxScore pruning for long posting lists
- [x] Fuzzing the file and log loaders
- [x] Collections: documents over dense, text and sparse sides, durable as a whole (`Collection`)
- [x] Automatic maintenance of collections (`MaintenancePolicy`, `maintain()`, a background thread)
- [x] Presets for collections (`CollectionOptions::preset`)
- [x] ACORN filtered graph search, chosen by the query planner
- [ ] RaBitQ compression
- [x] Python bindings with NumPy and pandas integration (`hnsw_lite_lib`)
- [x] Python wheels for Linux, macOS and Windows (`pip install hnsw-lite-lib`)
- [ ] C++ packaging, a C API (see the [suggested order](feature-catalog.md#suggested-order))
- [x] Search features: metadata, filtered search with a query planner, payload index (counts), batch search, range search
- [x] Payload index posting lists; faceted search (top values, numeric ranges, distinct counts)
- [ ] Filter-aware graph construction
- [x] Graph reordering for memory locality (`reorder()`), score modifiers (recency decay, confidence, boosts), dates, MMR diversity, document expiry (TTL), string keys
- [x] Benchmark tools: a comparison with hnswlib, FAISS and USearch, a stress test at a million vectors and more, a nightly job publishing charts
- [x] Half-precision vectors (float16, bfloat16), a compact ID map, and collection settings (expiry, string keys) saved with the collection
- [ ] Benchmark results on SIFT1M and GloVe, published
- [x] Collections: scanning every document, and exporting to a DataFrame (`scan()`, `documents()`, `to_pandas()`)
- [x] Faceted search: counts by field over the top hits and the whole corpus, filtered counts, numeric ranges
- [x] A command-line tool: `inspect`, `verify`, `health` and `export` for any file
- [ ] A first release on PyPI (v0.1.0), with a changelog
- [ ] LangChain and LlamaIndex integrations; an ann-benchmarks adapter
- [ ] Integer SIMD kernels for SQ8 distances; half precision for the disk index

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
- J. Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019.
- G. V. Cormack, C. L. A. Clarke and S. Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009.
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)