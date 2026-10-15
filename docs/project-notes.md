# Project notes

Why hnsw-lite exists, its current limitations, the roadmap, how it compares with established libraries, and references.

## Why this project

Vector search is the backbone of modern AI infrastructure: retrieval-augmented generation (RAG), semantic search and multimodal retrieval all depend on it. Production vector databases get their speed from low-level C++ work: memory layout, cache efficiency and SIMD math.

hnsw-lite implements that core in a small, readable codebase, with every design decision documented.

## Limitations

These are deliberate for the current stage:

- **Changes run one at a time.** Indexes are thread-safe, and adds, updates and removes take the lock alone. `add_batch` links its vectors on several threads, but searches wait while it runs; letting searches run during inserts would need the lowest storage layer redesigned for concurrent readers.
- **Heavy update churn gradually weakens the graph.** Each update links a new version and repairs around the old one, and these small losses add up: after 320 updates per vector on a 20-vector index, one or two vectors were unreachable from the entry point in 27 of 200 random update orders (worst 90% reachable), single-threaded as well as with threads. `compact()` rebuilds the graph and restores it fully, so call it after heavy update workloads. (hnswlib has the same weakness with repeated updates.)
- **Cosine indexes use twice the memory for vectors**, keeping each vector as given (for `get_vector`) besides the normalized copy used for searching.
- **Many exact copies of the same vector weaken the graph,** especially combined with heavy removal and update churn: graph reachability drops noticeably (hnswlib shares this limitation).
- **HNSW build speed.** One-by-one adds are single-threaded (use `add_batch` to build on several threads), and every access is bounds-checked for safety. Unchecked accessors in the hot loops are future optimizations.
- **Saving cannot force data onto the physical disk:** a save is atomic (temporary file, then rename), but a power cut right after `save` returns could lose that latest save. Loading is not memory-mapped, so a large index is read fully into memory.
- **HNSW reclaims memory by reuse.** A removed vector's slot is reused by the next insert, so memory stays bounded by the peak number of live vectors, but it is not returned to the system until `compact()` is called.
- **User IDs are 64-bit integers only.**
- **Filters are limited to 256 levels of nesting.** Combine many conditions with `in()`, `has_any_tag()` or a balanced tree instead.
- **Dynamic metadata fields are never removed**, even when no vector uses them: field names are the schema, and removing one automatically would silently forget its type. Unused strings, by contrast, are reclaimed.
- **The payload index keeps counts, not posting lists.** It makes the planner's estimates exact, but exact filtered search still checks every vector's metadata (a fast column read) rather than reading only the matching ones.
- **Filter-aware graph construction is not implemented.** Very selective filters go to exact search; between roughly 1% and 10% selectivity the planner's thresholds matter most, and may need tuning per dataset.
- **HNSW range search is approximate**, like k-nearest search; use `Strategy::ForceExact` when every vector in the radius must be found.
- **Custom predicates must be thread-safe** when used with batch search, and the planner can only estimate them by sampling.
- **Exact duplicates cannot all stay reachable.** Each node has a fixed number of link slots, and once one copy of a point is linked, further copies add nothing, so some copies become unreachable (with 200 identical vectors, about 60% stay reachable). They no longer harm other vectors, but deduplicate data if every copy must be returned.
- **Inner product is harder than L2 and cosine.** It is not a true distance, so recall is lower on un-normalized data (about 0.86 to 0.90 at `ef = 100` in tests, versus 0.997 or higher for L2 and cosine), and a small fraction of short vectors may become unreachable. This matches hnswlib. Use cosine, or normalize vectors, when possible.
- **Very small `M` builds a sparse graph.** `M = 2` works but leaves about 10% of nodes unreachable in tests; use `M >= 8`.
- **Kernels compare one pair of vectors at a time.** Batched query-vs-many kernels and matrix-multiplication-based flat search are possible future optimizations.
- **On some older Intel CPUs, heavy AVX-512 use lowers the clock speed.** If the AVX2 version benchmarks faster on your machine, that is why.

## Roadmap

Every feature that could come next, consolidated and in a suggested order, is in the [feature catalog](feature-catalog.md).

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
- [x] Search features: metadata, filtered search with a query planner, payload index (counts), batch search, range search
- [ ] Payload index posting lists, and filter-aware graph construction
- [ ] Benchmarks on SIFT1M and GloVe

The core capabilities (real deletion, updating vectors, saving and loading, thread safety and parallel building, all done) are planned in [docs/core-capabilities-plan.md](core-capabilities-plan.md), with 119 test scenarios in [docs/core-capabilities-test-plan.md](core-capabilities-test-plan.md). The [hardening and performance plan](hardening-and-performance-plan.md) schedules the remaining work together with edge-case fixes and optimizations.

## Comparison with existing implementations

| | hnsw-lite | hnswlib | Faiss | Qdrant | Milvus |
|---|---|---|---|---|---|
| Language | C++20 | C++11 | C++ (Python bindings) | Rust | Go and C++ |
| Scope | Learning-focused engine | HNSW library | Vector search library | Vector database | Distributed vector database |
| Vector layout | Separate aligned rows | Interleaved with level-0 links | Separate flat storage | Segments, memory-mapped | Via its Knowhere engine |
| Upper-level links | One arena allocation per node | One `malloc` per node | One shared array with offsets | Own implementation | Via Knowhere |
| SIMD selection | Runtime CPU detection | Compile-time flags with runtime checks | Separate builds per instruction set | Runtime CPU detection | Via Knowhere |
| Deletion | Real deletion: slot reuse and graph repair | Tombstones (optional slot reuse) | Depends on index type; not for HNSW | Yes | Yes |
| Quantization | No | No | Yes (PQ, SQ, IVF) | Yes | Yes |
| Metadata filtering | Yes, with a query planner | Limited (a filter function) | Limited (ID selectors) | Yes | Yes |
| Persistence | Not yet | Yes | Yes | Yes | Yes |
| Distribution across machines | No | No | No | Yes | Yes |

hnsw-lite focuses on the in-memory core that these systems share, and leaves out the database features built around it.

## References

- Yu. A. Malkov and D. A. Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv:1603.09320.
- [hnswlib](https://github.com/nmslib/hnswlib)
- [Faiss](https://github.com/facebookresearch/faiss)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Milvus](https://github.com/milvus-io/milvus)
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)