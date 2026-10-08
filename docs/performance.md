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

### Build type

Always measure Release builds. Debug builds are many times slower.