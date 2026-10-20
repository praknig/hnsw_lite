# Performance and tuning

Measured speed and recall, and how to choose settings for your data.

## Benchmarks

### Distance kernels

`bench/bench_distance.cpp` measures nanoseconds per squared-L2 call for each supported version. The vectors fit in the CPU cache, so this measures the math itself rather than memory speed. Build in Release mode before running it.

Example results on an x86-64 machine with AVX-512 (your numbers will differ):

| Dimension | Scalar | AVX2 | AVX-512 |
|---|---|---|---|
| 128 | 54.8 ns | 8.0 ns (6.8x) | 6.9 ns (7.9x) |
| 768 | 446.6 ns | 48.5 ns (9.2x) | 36.3 ns (12.3x) |
| 1536 | 905.4 ns | 102.1 ns (8.9x) | 72.7 ns (12.5x) |

The speedups come from two sources: wider instructions, and the 4 independent running sums, which the scalar loop cannot use because the compiler is not allowed to reorder float additions on its own.

### Search: Flat vs. HNSW

`bench/bench_search.cpp` builds both indexes on clustered synthetic data, uses `FlatIndex` results as ground truth, and measures queries per second and recall@10 for several `ef` values.

Example results: 50,000 vectors, 128 dimensions, L2, `M = 16`, `ef_construction = 200`, one CPU core with AVX-512 (your numbers will differ):

| Index | ef | Queries/s | Recall@10 | Speedup vs. Flat |
|---|---|---|---|---|
| Flat | - | 672 | 1.000 | 1.0x |
| HNSW | 10 | 40,965 | 0.788 | 61.0x |
| HNSW | 20 | 29,619 | 0.921 | 44.1x |
| HNSW | 40 | 13,749 | 0.984 | 20.5x |
| HNSW | 80 | 9,156 | 0.998 | 13.6x |
| HNSW | 160 | 7,059 | 0.999 | 10.5x |
| HNSW | 320 | 5,057 | 1.000 | 7.5x |

Build time: 0.02 s for Flat, 8.6 s for HNSW.

`ef` is the speed–accuracy dial: small values are very fast but miss some neighbors; around `ef = 80` HNSW is over 13x faster than exact search while finding 99.8% of the true neighbors. The gap grows with the number of vectors, because Flat's cost grows linearly while HNSW's grows roughly logarithmically.

## Baseline for performance work (Phase 0)

Measured with `bench_baseline` (in [bench/](../bench)), which reports build time, index memory, and for `ef` from 16 to 256: queries per second (the median of 5 passes), p50 and p99 latency, and recall@10. hnswlib (v0.10.0 release candidate, header-only C++) was measured with identical data, settings (`M = 16`, `ef_construction = 200`, k = 10) and method.

**Data:** 200,000 vectors of 128 dimensions, L2, 1,000 queries. Points come from 100 clusters in a 16-dimensional space, mapped into 128 dimensions with a little noise, because real embeddings vary along far fewer directions than their dimension. (Plain random noise in all 128 dimensions makes nearest neighbors nearly meaningless: both libraries then reach only 15% to 57% recall, so such data says nothing about either.)

**Machine:** one shared virtual core (Intel Xeon, 2.1 GHz), AVX-512, GCC, Release build, single-threaded.

| | hnsw-lite | hnswlib |
|---|---|---|
| Build | 40.2 s (4,972 vectors/s) | 40.5 s (4,937 vectors/s) |
| Index memory | 151 MiB | 144 MiB |

| `ef` | hnsw-lite queries/s | hnsw-lite recall@10 | hnswlib queries/s | hnswlib recall@10 |
|---|---|---|---|---|
| 16 | 37,922 | 0.953 | 32,290 | 0.955 |
| 32 | 20,667 | 0.993 | 14,603 | 0.994 |
| 64 | 11,309 | 1.000 | 11,457 | 1.000 |
| 128 | 6,542 | 1.000 | 6,315 | 1.000 |
| 256 | 3,435 | 1.000 | 3,828 | 1.000 |

**Reading these numbers:** the two libraries are level. Build time and recall match, and search speed is within the machine's noise at every `ef`; hnsw-lite uses about 5% more memory.

**Measurement noise:** on this shared machine, running the same program three times gave 18,759 to 24,773 queries per second (`ef = 64`, 50,000 vectors) and builds of 5.7 to 6.6 seconds: differences of up to 30%. That is far above the 5% gain an optimization must show to be kept, so **optimizations are judged on a quiet machine with dedicated cores**, with the baseline and the candidate run alternately several times and their medians compared.

### Running the baseline on real data

`bench_baseline --files` reads the standard `.fvecs` and `.ivecs` formats. For SIFT1M (1 million vectors of 128 dimensions, about 160 MB to download), on Windows:

```powershell
curl.exe -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
.\build\bench_baseline.exe --files sift\sift_base.fvecs sift\sift_query.fvecs sift\sift_groundtruth.ivecs l2 1000
```

Add `--threads=N` to any of these commands to build with `add_batch` on N threads (`--threads=0`: all cores) instead of adding vectors one by one; the header line shows which was used. The ground-truth file holds the exact nearest neighbors, so recall is measured without a brute-force search over a million vectors. Building the index takes several minutes on one thread.

## Tuning guide

### Flat or HNSW?

`FlatIndex` costs one distance computation per stored vector per query, so its speed falls in proportion to the collection's size: about 670 queries per second at 50,000 vectors of 128 dimensions on one core (table above). `HnswIndex` visits only a few hundred vectors per query, so it stays fast as the collection grows, at the cost of occasionally missing a true neighbor.

Use Flat when results must be exact, when the collection is small (up to about 10,000 vectors, it is fast enough and needs no building), or as ground truth to measure HNSW's recall on your own data. Use HNSW otherwise.

### HNSW build settings

| Setting | Default | Raise it to... | Lower it to... | Typical range |
|---|---|---|---|---|
| `M` | 16 | Improve recall on hard data (high dimensions, clustered data) | Save memory: each level-0 node uses `2 * M * 4` bytes of links | 8 to 48 |
| `ef_construction` | 200 | Build a better graph | Build faster | 100 to 400 |
| `repair_on_remove` | true | Keep recall high under heavy removal | Remove faster, if removals are rare or `compact()` is run often | |

Values of `M` below 8 leave parts of the graph poorly connected (at `M = 2`, tests find about 10% of vectors unreachable).

### The search setting: `ef`

`ef` is the number of candidates a search keeps. It must be at least `k`, and is raised to `k` automatically. From the benchmark above: `ef = 40` finds 98.4% of true neighbors, `ef = 80` finds 99.8%, and each doubling roughly halves speed beyond that. Measure on your own data by comparing HNSW results with `FlatIndex` results for a few hundred queries, and choose the smallest `ef` that meets your recall target.

### Metric

Cosine and L2 give the best HNSW recall. Inner product is not a true distance, so recall is lower on vectors of very different lengths (0.86 to 0.90 at `ef = 100` in tests, against 0.997 or more for L2 and cosine). If your model's vectors can be normalized, normalize them and use cosine.

### Filters

- The planner's defaults (exact scan for at most 2,000 matches or 1% of vectors) suit most data. On very large indexes, lower `max_exact_fraction`: 1% of 100 million vectors is a million distance computations.
- Call `create_payload_index(field)` for keyword, boolean or tag fields you filter on often, so the planner counts matches exactly instead of estimating them.
- Prefer `in` and `has_any_tag` over long chains of `||`: they are evaluated in one step.
- `SearchStats` shows what the planner chose and how much work the search did; use it to check a slow query.

### Removals and `compact()`

Removed HNSW slots are reused by later inserts, so memory stays bounded by the peak number of vectors. Graph repair keeps recall high, but after a large share of the index has been replaced (say a third), `compact()` restores the best graph quality and returns memory. It takes about as long as building the index.

### Batch search

`search_batch` with `threads = 0` uses one thread per CPU core. For `FlatIndex`, batches are also faster per query on one thread, because each stored vector loaded into the cache is compared with 16 queries at once.

### Compression

Measured recall@10 on synthetic clustered data (see the `sq8`, `binary`, `pq` and `codes_only` test groups). Code distances use SIMD kernels (AVX2, AVX-512, NEON). `bench_codecs [dimension]` times them against full-precision distances. Measured on a Windows desktop PC (AVX2, MinGW GCC, Release build), 768 dimensions, nanoseconds per distance:

| Kernels | Float L2 | SQ8 L2 | SQ8 dot | Binary |
|---|---|---|---|---|
| Plain C++ | 467 | 471 | 459 | 4,169 |
| AVX2 | 149 | 93 | 79 | 99 |

With AVX2, SQ8 distances were 1.6 to 1.9 times faster than full-precision ones and binary distances 1.5 times faster, while reading 4 and 32 times less memory; binary distances without SIMD were about 9 times slower than floats (the AVX2 kernel is 42 times faster). An earlier run on a shared cloud machine with AVX-512 gave a similar picture (floats 115 ns; SQ8 70 and 57 ns; binary 51 ns). Whole-search speed has not been measured yet.

| Setting | Recall@10 | Memory for vectors |
|---|---|---|
| Full precision (HNSW, `ef` 64) | 0.93 to 0.99 | 4 bytes per dimension |
| SQ8, re-ranked (default) | the same as full precision | 5 bytes per dimension (codes and floats) |
| SQ8, codes only (`keep_full_vectors = false`) | 0.972 to 0.976 (about 1.5% lower) | 1 byte per dimension |
| Binary, re-ranked, Flat `rerank_factor` 32 or HNSW `ef` 400 | 0.96 to 0.99 | 4 bytes and 1 bit per dimension |
| Binary, re-ranked with the defaults (`rerank_factor` 4, `ef` 64) | 0.63 to 0.79 | the same |
| Binary, codes only | 0.34 to 0.39: not recommended | 1 bit per dimension |
| PQ (8 dimensions per part), re-ranked: Flat `rerank_factor` 16, HNSW `ef` 200 | 0.89 to 0.97 | 4 bytes per dimension plus 1 byte per part |
| PQ, codes only | 0.31 to 0.40 on this test data | 1 byte per part (32 times smaller by default) |

- **SQ8** is the safe choice: re-ranked, it matches full precision; with codes only it saves three quarters of the vector memory for about 1.5% recall.
- **Binary codes** need a large re-ranking budget: a `rerank_factor` of 16 to 32 for Flat, a larger `ef` for HNSW. Keep the full vectors with them.
- **PQ** gives the smallest codes for a given recall once re-ranked with a generous budget; choose more parts (`pq_subspaces`) for better recall. Its training (k-means) is slower than SQ8's.
- **Matryoshka truncation** (`truncate_to`) cuts the cost of every distance in proportion, for embeddings trained for it; add `keep_full_vectors = false` to save the memory too.
- With codes only, training is final: train on a representative sample (`train_after`).

### Inverted file (IVF)

Measured on noisy synthetic data (5,000 vectors of 32 dimensions in 40 clusters, 64 lists; test `i04`):

| Probes | Vectors scanned | Recall@10 |
|---|---|---|
| 1 | 1.5% | 0.16 |
| 4 | 6% | 0.40 |
| 16 | 25% | 0.81 |
| 64 (all) | 100% | 1.00 (exact) |

Scan time grows with the share scanned, so probes trade speed for recall; real embeddings cluster better than this data and usually need fewer. IVF-PQ (32 lists, PQ re-ranked with `rerank_factor` 16) reached 0.56 at 8 probes and 0.87 with all lists probed, the same as PQ alone. Measure on your data: start with `lists` near the square root of the number of vectors and raise `probes` until recall suffices.

### Disk-based index

Measured on 5,000 clustered vectors of 32 dimensions (R 32; tests `dk09` and `dk15`; reads counted, not timed):

| `L` (W 4) | Nodes read per query | Rounds | Recall@10 |
|---|---|---|---|
| 20 | 30 (0.6%) | 9 | 0.86 |
| 50 | 56 (1.1%) | 16 | 0.98 |
| 100 | 105 (2.1%) | 27 | 1.00 |

Each node read is one 4 KB block. `W` 1 needed 52 rounds at `L` 50 for the same reads: on storage with high latency, a larger `W` cuts waiting, and with `DiskOpenOptions::io_threads` the reads of a round also overlap. With a simulated 3 ms per read and W 4, five searches (60 rounds) took 679 ms with reads one after another and 194 ms with 4 reading threads (test `dk26`): about one latency per round, 3.5 times faster. Search time is roughly reads times the storage's latency (unless cached by the operating system), so measure on your storage. The navigation codes matter: with one PQ part per 8 dimensions instead of 2, recall at `L` 50 fell from 0.98 to 0.71 for the same reads.

### Durability

`bench_durability [count] [directory]` times additions without durability and in each mode. Measured on a virtual disk (2,000 additions of 64 dimensions): not durable 10,013 per second, `Buffered` 9,726, `Grouped` 8,863, `Synced` 4,222. The difference is the cost of a flush, which on real disks is 0.1 to several milliseconds: `Synced` can then be 10 to 50 times slower, while `Grouped` stays close to `Buffered`. Run it on the disk you will use.

### Build type

Always measure Release builds. Debug builds are many times slower.
