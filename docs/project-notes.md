# Project notes

Why hnsw-lite exists, its current limitations, the roadmap, how it compares with established libraries, and references.

## Why this project

Vector search is the backbone of modern AI infrastructure: retrieval-augmented generation (RAG), semantic search and multimodal retrieval all depend on it. Production vector databases get their speed from low-level C++ work: memory layout, cache efficiency and SIMD math.

hnsw-lite implements that core in a small, readable codebase, with every design decision documented.

## Limitations

These are deliberate for the current stage:

- **One writer at a time.** `add` and `remove` must not run alongside anything else. Concurrent searches are safe when no writer is active. Concurrent inserts will need per-node locks and an atomic arena.
- **HNSW build speed.** Building is single-threaded, and every access is bounds-checked for safety. Parallel construction and unchecked accessors in the hot loops are future optimizations.
- **No persistence.** Data lives only in memory. The arena stores raw pointers; switching to offsets will make saving to disk straightforward.
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

- [x] **Layer 1:** aligned vector store, arena allocator, graph storage, ID mapping, tests
- [x] **Layer 2:** SIMD distance kernels (scalar, AVX2, AVX-512, NEON), runtime dispatch, tests, benchmark
- [x] **Layer 3a:** Flat index with top-k heap, used as ground truth
- [x] **Layer 3b:** HNSW index: random levels, insertion, neighbor heuristic, beam search, visited lists
- [x] Search benchmark on synthetic data (recall@10 vs. queries per second)
- [x] Continuous integration: Linux (GCC, Clang, sanitizers), Windows (MSVC, MinGW) and macOS on ARM
- [x] Real deletion: reusable IDs, slot reuse, graph repair, `compact()`
- [ ] Updating vectors: `update` and `upsert`
- [ ] Saving and loading indexes
- [ ] Concurrent inserts, including parallel HNSW construction (`add_batch`)
- [x] Search features: metadata, filtered search with a query planner, payload index (counts), batch search, range search
- [ ] Payload index posting lists, and filter-aware graph construction
- [ ] Benchmarks on SIFT1M and GloVe

The core capabilities (real deletion, now done, then updating vectors, saving and loading, and concurrent inserts) are planned in [docs/core-capabilities-plan.md](core-capabilities-plan.md), with 119 test scenarios in [docs/core-capabilities-test-plan.md](core-capabilities-test-plan.md).

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