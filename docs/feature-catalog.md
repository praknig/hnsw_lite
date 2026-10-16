# Feature catalog

Every feature that could be added to hnsw-lite, in one place: the open roadmap items, the open items of the [hardening and performance plan](hardening-and-performance-plan.md) (N1 to N10, decisions D3 to D5, the Phase 2 quick wins), the testing ideas, and new features. Overlapping ideas are merged into one entry; the **Merges** column records what each entry absorbed, so nothing is lost.

**Status:** *roadmap* means already on the [roadmap](project-notes.md#roadmap) or in a plan; *new* means proposed here. **Effort** is a rough estimate for one developer, including tests and documentation.

## Contents

1. [Records and data access](#1-records-and-data-access)
2. [Search](#2-search)
3. [Filtering](#3-filtering)
4. [Memory and index types](#4-memory-and-index-types)
5. [Storage and durability](#5-storage-and-durability)
6. [Concurrency](#6-concurrency)
7. [Health and maintenance](#7-health-and-maintenance)
8. [Performance](#8-performance)
9. [Interfaces](#9-interfaces)
10. [Quality and portability](#10-quality-and-portability)
- [Left out deliberately](#left-out-deliberately)
- [Suggested order](#suggested-order)

## 1. Records and data access

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 1.1 | **Read back records:** `get_vector(id)`; list and page through all records, optionally filtered; `count(filter)` | **done** (`get_vector`, `get`, `scan`, `count_where`) | A gap today: a stored vector can only be read through internal storage | 1 day |
| 1.2 | **Bulk changes:** `remove_batch`, plus removing and updating metadata by filter (`remove_where`, `set_metadata_where`) | **done** | Delete or update by filter | |
| 1.3 | **String and UUID IDs**, alongside 64-bit numbers | new | | 2 days |
| 1.4 | **Multi-vector records:** several vectors per record, either named (for example title and body) or as chunks of one document, with results grouped per record | new | Named vectors; several vectors per document | 4–5 days |
| 1.5 | **Expiry (TTL):** records removed automatically after a set time | new | | 1 day |

## 2. Search

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 2.1 | **Search by stored ID:** "more like record 42", without fetching its vector first | **done** (`search_by_id`) | | Half a day |
| 2.2 | **Recommendations:** several positive and negative examples, given as vectors or IDs (as in Qdrant) | new | Builds on 2.1 | 1–2 days |
| 2.3 | **Diversity re-ranking (MMR):** results that are relevant and different from each other | new | | 1 day |
| 2.4 | **Hybrid search:** keyword matching (BM25) and sparse vectors (such as SPLADE), merged with dense results by rank fusion | new | Hybrid keyword search; sparse vectors | 1–1.5 weeks |
| 2.5 | **Deadlines and cancellation:** for searches (returning the best results found so far) and for long operations (`add_batch`, `compact`, `save`), with progress reports | **done** | Search deadlines; progress and cancellation | 1–2 days |
| 2.6 | **Automatic ef:** choose `ef` per query or per index to reach a target recall | new | | 1–2 days |
| 2.7 | **More metrics:** Hamming for binary vectors, Manhattan (L1) | new | | 1–2 days |

## 3. Filtering

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 3.1 | **New field kinds:** geographic points (within a radius or a bounding box), dates and times, text prefix and contains | new | Geographic filters; date fields; text matching | 3–4 days |
| 3.2 | **Facet counts:** the number of matches per value, for building filter menus | new | Builds on the payload index, which already counts values | 1 day |
| 3.3 | **Faster filtered search:** payload posting lists, reusable compiled filters, cached selectivity estimates | roadmap | Posting lists (roadmap); N2; N3 | 3 days |
| 3.4 | **Filter-aware graph and multi-tenancy:** high recall for selective filters, including a tenant field that isolates each customer's data | roadmap + new | Filter-aware graph construction (roadmap); multi-tenancy | 1–1.5 weeks |

## 4. Memory and index types

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 4.1 | **Quantization:** 8-bit (SQ8), binary and product quantization (PQ), each optional per index and with optional re-ranking at full precision. Needs file format version 3 (version 2 added cosine originals). **Done:** SQ8, binary and PQ, with SIMD kernels; see the [compression and index types plan](compression-and-index-types-plan.md) | **done** | SQ8 (roadmap, decision D4: an option); binary quantization; PQ | 3–4 days for SQ8; 2–3 more for binary; 1 week more for PQ |
| 4.2 | **Shorter vectors (Matryoshka truncation):** keep only the first N dimensions of embeddings trained for it | **done** ([plan](compression-and-index-types-plan.md)) | | 1 day |
| 4.3 | **An IVF index:** a second index type, for very large datasets and fast builds | **done**, as an option of `FlatIndex` (IVF-PQ with PQ; [plan](compression-and-index-types-plan.md)) | | 1 week |
| 4.4 | **A disk-based index (DiskANN-style):** indexes larger than memory. **Planned** ([plan](compression-and-index-types-plan.md)), read-only at first | new | | 4–6 weeks (re-estimated) |

## 5. Storage and durability

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 5.1 | **Durability:** a write-ahead log, so changes since the last save survive a crash, plus snapshots | new | | 3–4 days |
| 5.2 | **Memory-mapped loading:** near-instant loading of large indexes; includes the arena storing offsets instead of pointers | roadmap | Memory-mapped loading; N8 (offsets) | 3 days |
| 5.3 | **Collections:** many named indexes in one database directory | new | | 2 days |
| 5.4 | **Import and export:** NumPy `.npy`, `.fvecs` and `.ivecs`, metadata as JSON | new | Reading `.fvecs` exists in `bench_baseline` | 1–2 days |

## 6. Concurrency

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 6.1 | **Searches during changes:** searches no longer wait for `add_batch` and other changes; needs storage that is stable for concurrent readers | roadmap | The deferred option of stage B (scenarios C25, C28, C29) | 1–1.5 weeks |
| 6.2 | **Thread pool and memory limits:** a stable pool when batch thread counts alternate, and documented or capped memory per concurrent search | roadmap | N6; N7 | 1 day |

## 7. Health and maintenance

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 7.1 | **Index health report:** memory breakdown, graph reachability, share of removed slots, update churn, with advice on when to compact | new | | 1 day |
| 7.2 | **Graph upkeep under churn:** better repair during updates, and an optional automatic `compact()` past a threshold | roadmap + new | The update-churn finding (see [project notes](project-notes.md#limitations)); N5 | 1–2 days |
| 7.3 | **Check and repair:** validate a loaded or running index, and repair what can be repaired | new | | 1 day |

## 8. Performance

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 8.1 | **Search quick wins:** prefetching, batched neighbor distances, allocation-free search, smaller visited marks, a cheaper duplicate check, fewer bounds checks | roadmap | Phase 2 (O1–O4, O8); N1. Each kept only if at least 5% faster on a quiet machine | 3–4 days |
| 8.2 | **Distance code options:** choose the kernel manually (for example AVX2 where AVX-512 lowers the clock speed), and an exact or fast matrix mode for Flat batch searches | roadmap | N4; D3 (decided: the user chooses) | 2–3 days |
| 8.3 | **Links stored next to vectors** | roadmap | Phase 4 layout decision | 1 week, measured |
| 8.4 | **Real-data benchmarks:** SIFT1M and GloVe, with hnswlib for comparison, and the parallel build speedup | roadmap | Phase 0 on a quiet machine; `bench_baseline --threads` | 1 day |

## 9. Interfaces

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 9.1 | **Python bindings** | roadmap | | 2–3 days |
| 9.2 | **C API:** a stable interface that Rust, Go, C# and other languages can call | new | | 2 days |
| 9.3 | **Command-line tool:** build, inspect, check (7.3), search, compact and convert (5.4) index files without writing code | new | Includes the user-facing parts of 5.4 and 7.3 | 1–2 days |
| 9.4 | **HTTP server:** a small REST service around the library | new | | 3–4 days |

## 10. Quality and portability

| # | Feature | Status | Merges | Effort |
|---|---|---|---|---|
| 10.1 | **Fuzzing the file loader** (scenarios F1 to F8): millions of malformed files, under the sanitizers, with every crash kept as a regression test | roadmap | Fuzzing (roadmap); scenarios F1–F8 | 1 day |
| 10.2 | **Nightly additions:** the AVX-512 kernel under Intel's emulator (A1–A3), and a million-vector build, save, load and search | new | AVX-512 check; large-scale nightly job | 1 day |
| 10.3 | **Platform coverage:** Windows on ARM (needs build changes first), and rejecting 32-bit and big-endian builds with a clear message | roadmap | Windows on ARM; N10 | 1–2 days |

## Left out deliberately

**Sharding, replication, encryption at rest and access control.** These belong to a database server such as Qdrant or Milvus, not to a library. The HTTP server (9.4) is as far as hnsw-lite should go in that direction.

## Suggested order

| Stage | Features | Why | Total effort | Progress |
|---|---|---|---|---|
| **A. Complete the basics** | 1.1, 1.2, 2.1, 2.5, 7.1, 7.2, 10.1 | Small gaps any user notices, the update-churn finding, and fuzzing the code most exposed to untrusted input | About 1.5 weeks | 1.1, 1.2, 2.1, 2.5 done; 7.1, 7.2, 10.1 to do |
| **B. Easier to adopt** | 9.1, 9.3, 5.4 | Python and a command-line tool open the library to far more people | About 1 week | Not started |
| **C. Smaller indexes** | 4.1 (SQ8 first), 5.2 | Memory is usually the first limit users reach | About 1.5 weeks | 4.1 done (SQ8, binary, PQ); 4.2 done too; 5.2 to do |
| **D. Better results for text** | 1.4, 2.4, 3.1 | What retrieval-augmented generation (RAG) and search applications need most | About 3 weeks | Not started |
| **E. Scale and speed** | 3.3, 3.4, 6.1, 8.1–8.4, 4.3 | Worth doing once there is real usage to measure | Several weeks | 4.3 done |

The remaining features (1.3, 1.5, 2.2, 2.3, 2.6, 2.7, 3.2, 4.4, 5.1, 5.3, 6.2, 7.3, 9.2, 9.4, 10.2, 10.3) fit in wherever a user asks for them.