# Testing and continuous integration

How hnsw-lite is tested: the test suite and its runner, code coverage, the CI pipeline, and the bugs testing has found.

## Testing

### The comprehensive suite

`tests/test_comprehensive.cpp` contains **every test scenario in one program: 745 individually named tests** in 36 groups, with a built-in runner. It needs no external test framework.

| Group | Tests | What it covers |
|---|---|---|
| `layer1` | 58 | `round_up`; AlignedBlock (alignment, fill, size rounding, moves); Arena (every alignment, exact fill, oversized requests, 100 arrays staying intact); VectorStore (strides, padding, stable addresses, block size limit, 10,000 vectors, special float values); IdMap (extreme IDs, unknown IDs, removed IDs staying taken); GraphStorage (capacities, level limits, `set_links` at, below and above capacity); Storage (staying in sync after every kind of rejected insert) |
| `layer2` | 31 | Dispatch consistency; every kernel on every supported CPU version: double-precision reference at 9 lengths, known values, length 0, zero vectors, overflow, NaN propagation, exact symmetry, unaligned input, padding, 8,192 dimensions, exact identities (`cos = 1 + ip`, L2 scaling by 4), versions agreeing; `normalize` on tiny, huge, zero, negative and empty vectors |
| `helpers` | 31 | Candidate ordering; TopK (ties, k = 0, 1 and 2⁶⁴−1, infinity, reuse, matching a full sort of 10,000 candidates); PreparedVector (NaN and ±∞ rejected while keeping old contents, padding after reuse, tiny and extreme values); VisitedList (growth, epoch wrap-around, 1,000 resets); VisitedListPool (reuse, return after exceptions, 8 threads) |
| `flat` | 24 | Exact match with a reference for all metrics and k up to 500; ties ordered by insertion; empty index, k = 0, k > size, huge k; wrong dimension, NaN, ∞ and duplicates rejected; removal; inner-product order; cosine with zero vectors; extreme IDs; 1,536 dimensions; 5,000 vectors |
| `hnsw` | 44 | Parameter and input validation; rejected inserts leaving the later graph bit-identical; 1 to 10 vectors matching Flat exactly; k and ef limits; results sorted, unique and live; distances equal to Flat's; determinism; graph validity and recall for every metric; recall at k = 1, 10 and 50; level distribution; `M = 2`, `ef_construction = 1`, `M = 64`, other seeds; identical vectors; removing the entry point, a quarter, all but one, and everything; new nodes never linking to removed ones |
| `robustness` | 20 | **Numeric extremes:** inner-product overflow (+∞ plus −∞ = NaN) sorting last instead of breaking the order, L2 overflow, huge values with cosine, denormals, −0 versus 0. **Out of memory:** every allocation in an `IdMap`, `VectorStore`, `GraphStorage` and `Storage` insert, a Flat add and search, and an HNSW first insert, insert and search is made to fail in turn, checking after each failure that nothing is corrupted and the object still works |
| `deletion` | 59 | Scenarios D1 to D66 from the [test plan](core-capabilities-test-plan.md), each test named after its scenario ID, plus 8 tests added by the coverage audit. Layer 1 building blocks (release, bind, overwrite, move, node reset); Flat swap-with-last removal checked against a `std::map` reference over 10,000 random operations; HNSW slot reuse at higher and lower levels, graph repair, entry-point reassignment, removing everything, 20-cycle churn for every metric, removing a whole cluster; `compact()`; statistics; out-of-memory sweeps for every new operation |
| `metadata` | 32 | MD scenarios: every type round-trips; strict and dynamic schemas; type errors; extreme values (64-bit limits, 1 MB strings, UTF-8, 1,000 tags); `set_metadata` and `unset`; metadata following slots through swap-with-last, reuse and `compact()`; 3,000 random operations against a reference; out-of-memory sweeps, including no field left behind by a failed insert |
| `filter` | 35 | FL scenarios: every condition, and 1,000 random nested filters, checked against an independent reference evaluator; missing-field and exact integer/float rules; type errors; Flat exact against brute force; HNSW never violating a filter across 1,000 random filters, recall at 50% to 5% selectivity, matches clustered or scattered, entry point failing the filter; predicates alone, combined, throwing, and called from 8 threads |
| `planner` | 18 | PL scenarios: sampled selectivity accuracy; exact payload counts; each threshold and its inclusive boundary; per-query overrides; forced strategies; exact strategy equal to Flat; tiny, empty and all-removed indexes; determinism; consistent statistics; invalid settings; recall at 1% to 100% selectivity |
| `payload` | 6 | PI scenarios: counts exact after 2,000 random operations; identical results with or without the index; backfill equal to incremental; unsupported types rejected; out of memory; kept by `compact()` |
| `batch` | 16 | BT scenarios: identical to single searches for every metric and thread count; tile boundaries; invalid input rejected before any work; filters and predicates; 10,000 queries; the thread pool under failures, stress and concurrent callers |
| `range` | 16 | RG scenarios: Flat exact for every metric; squared L2 radius; radius 0, negative, infinite and NaN; optional cap; HNSW recall by radius; crossing outside vectors; filters; removed vectors; forced exact equal to Flat |
| `search_e2e` | 6 | IX scenarios: a 4,000-step lifecycle with filters and `compact()`; batches during churn; 8 threads; every metric with every feature; reproducibility; 20,000 vectors with random filters |
| `update` | 32 | Scenarios U1 to U20 from the [test plan](core-capabilities-test-plan.md), each named after its ID: Flat and HNSW `update` and `upsert`, unknown IDs, invalid input, the entry point, every vector updated, one ID updated 1,000 times, small and large moves, copies of other vectors, a one-vector index, every metric, reproducibility, and an out-of-memory sweep proving the old vector survives every failure. Plus U21 to U27: metadata kept or replaced, invalid metadata, the new ID-map operations, updates without graph repair, 5,000 mixed operations against a reference, and five edge cases: a zero vector with cosine, `upsert` with an invalid vector for a new ID, `compact()` after many updates, the dictionary cleanup during an update, and an updated vector never left without links |
| `hardening` | 14 | Edge cases E1 to E13 of the [hardening and performance plan](hardening-and-performance-plan.md): random levels identical on every platform, nested batches from predicates, changes during a search refused, cosine distances never negative, the `M` limit, extreme IDs, vectors taken from the index itself, dimension 0, thousands of fields and filter tags, throwing predicates in threaded batches, subnormal values, the deepest filter on pool threads, and the change detection released on every path |
| `persistence` | 35 | Scenarios P1 to P42 from the [test plan](core-capabilities-test-plan.md), each named after its ID: round trips for every metric and index state (empty, one vector, removals, updates, compaction, extreme IDs, several dimensions), settings preserved, continuing after a load building the identical graph, deterministic bytes, safe overwriting, a file truncated at every byte and a bit flipped in every byte, wrong headers and kinds, extra bytes, inflated counts, invalid content behind valid checksums, unusable paths, out-of-memory sweeps of loading and saving, golden files reproduced byte for byte, and a 100,000-vector round trip. Plus P50 to P57: metadata, every search feature after loading, changes continuing identically, saving during a search, unseekable streams, metadata damage, failing streams and every validation rule |
| `add_batch` | 19 | Scenarios B1 to B19: 1 to 8 threads giving graphs as good as one-by-one adds; one thread byte-identical to one-by-one adds (every metric, with and without metadata, into removed slots, into an index with every vector removed); duplicate IDs, NaN, infinity, wrong sizes, invalid or inconsistently typed metadata changing nothing; empty batches; reused slots; entry point and top level with many high-level vectors racing; searches seeing all of a batch or none; refusal inside a search; out of memory at every point, on 1 and 4 threads, into a filled and an empty index, with and without graph repair (no batch vector remains, earlier vectors intact and searchable); payload counts; save and load; later changes; re-added IDs; Flat restored exactly; batches mixed with single adds; 8 threads with every metric |
| `bulk` | 16 | Scenarios W1 to W16 for `remove_where`, `set_metadata_where` and `count_where`, on HNSW and Flat: exactly the matching vectors removed or changed (checked against expected matches computed independently for eight kinds of filter); no match changing nothing; everything matching; unknown fields under strict and dynamic schemas; removing 30% (including the entry point, and with graph repair off) with the graph healthy and the IDs reusable; payload counts; only the given fields changed; invalid changes rejected even when nothing matches; changing the filtered field itself; out of memory at every point (removal completes, metadata changes all or nothing: confirmed to fail against an in-place version); refusal inside a search; searches seeing all of a change or none; save and load; later changes; Flat moving metadata with vectors |
| `records` | 17 | Scenarios R1 to R17 for `get_vector`, `get` and `scan`: vectors read back bit for bit for every metric (cosine originals included); paging through everything with several page sizes, each ID once and ascending; filtered paging for eight kinds of filter; edge cases (empty, exact fit, limit 0, cursors beyond the end or on removed IDs, extreme IDs); records without vectors or metadata; changes between pages; paging while another thread changes the index; unknown filter fields; out of memory; reads inside a search; save and load; version 1 cosine files; originals following every path (confirmed to catch a missing move and a compaction from normalized vectors); damaged cosine files |
| `control` | 25 | `remove_batch` (K1 to K6): exactly the listed IDs, unknown and repeated IDs, empty lists, everything, out of memory, refusal inside a search. `search_by_id` (S1 to S6): the same as searching with the stored vector (every metric), including or excluding the record, filters and predicates, unknown IDs, the largest k, duplicates of the record, atomic while another thread updates it. Deadlines, cancellation and progress (T1 to T13): searches of every kind stop early with valid results (checked by work done, not timing), a cancel from another thread, no change without a deadline, a nested search's own deadline; `add_batch`, `compact` and `save` cancelled at the start, middle and end, by flag, deadline or a throwing progress callback, always undone; progress in order; cancelling 8 linking threads |
| `codec` | 16 | Phase Q0: k-means finding separated clusters; bit-identical results across runs and thread counts; a fixed answer every platform must reproduce; k equal to the number of points, k of 1, identical points, duplicates with many centroids; invalid input; out of memory. SQ8: round-trip error at most half a level; constant dimensions stored exactly and values outside the range clamped; distances matching the decoded vectors for every metric; ranking close to full precision; fixed-answer codes (expected values computed independently with NumPy float32); misuse rejected; restoring a trained range; the codec factory. The SIMD code kernels: every instruction set the CPU supports against plain C++, at every length from 0 to 130 and unaligned addresses, and the binary bit order one bit at a time (a reversed order was shown to be caught) |
| `sq8` | 12 | Phase Q1: untrained means full precision (identical results); automatic training at the threshold (single adds and batches) and explicit training (with its errors); recall for every metric with and without re-ranking, HNSW and Flat; codes following every path (removals, reuse, updates, upserts, batches, a cancelled batch, compaction); saving and loading byte for byte, trained or not; every truncation and flipped bit of SQ8 files, and seven invalid values behind valid checksums (each confirmed to be rejected by its own rule); batch, filtered and exact searches with codes, range searches at full precision; out of memory in training and adding; searching while another thread trains; older files without a codec; invalid settings |
| `matryoshka` | 11 | Phase Q2: identical results (IDs and distances) to an ordinary index of the first N values, for every metric, HNSW and Flat (confirmed to fail if the wrong dimensions are kept); vectors read back in full by `get_vector`, `get`, `scan` and `search_by_id`; inputs checked in full (wrong sizes, NaN in the part cut off); settings (the full dimension means off, a larger truncation is rejected, a single dimension works); with SQ8; saving and loading byte for byte; every slot path keeping full vectors; every truncation and flipped bit of a file, and a truncation larger than the dimension; batch and range searches; truncation without a codec; out of memory |
| `lean` | 8 | `keep_full_vectors = false`: truncating indexes give identical results to keeping full vectors and return the stored first N values (every metric, HNSW and Flat); cosine indexes return normalized vectors; every slot path, with `search_by_id` and `compact` rebuilding inputs from stored vectors; saving and loading byte for byte, with smaller files; damaged files and an invalid flag; a flipped flag detected; with SQ8; the default keeps everything |
| `review` | 4 | A review of phases Q0 to Q2 and `keep_full_vectors`: a range search inside a coded search's predicate (checked on every call); a huge `rerank_factor`; a **model-based test**: 24 random configurations (metric, dimension, truncation, SQ8 trained automatically or explicitly, re-ranking, keeping full vectors), 300 random operations each (adds with duplicate and very large vectors, removals, updates, upserts, batches including refused ones, `remove_batch`, `compact`, training, `search_by_id`, saving and reloading), every one followed by a full check against a simple model (sizes, every vector read back, exact HNSW against Flat up to the order of ties, codes, byte-identical saving); and compaction never changing stored vectors |
| `codes_only` | 8 | SQ8 keeping only codes: floats kept until training and dropped after it, with recall for every metric (HNSW and Flat) and vectors read back within half a step; every slot path (removals, reuse, updates, upserts, batches, a cancelled batch, `compact`) and every kind of search from codes; saving and loading byte for byte, with a much smaller file; training again refused; every truncation and flipped bit of a file, and three inconsistent files behind valid checksums (each confirmed rejected by its own rule); searching while another thread trains and drops the floats; out of memory (training all or nothing, floats kept on failure); with truncation and cosine |
| `binary` | 12 | Phase Q3: code sizes and bit order; a fixed answer every CPU must reproduce (expected values computed independently with NumPy float32); distances matching decoded vectors for every metric; decode then encode exact (constant dimensions included); misuse rejected; recall with and without re-ranking, by `rerank_factor` and `ef`, and with codes only; every slot path, keeping full vectors or not; saving and loading byte for byte; every truncation and flipped bit of a file, and a negative spread; with truncation; training final with codes only; out of memory. A golden file with binary codes (`golden_hnsw_binary_v3.bin`) must be reproduced on every platform |
| `pq` | 12 | Phase Q4: parts and code sizes (uneven parts, too many parts refused); fewer training vectors than centroids stored exactly; distances matching decoded vectors for every metric; decode then encode exact with duplicate centroids; misuse rejected (invalid states, codes naming missing centroids); recall by re-ranking budget and with codes only; every slot path, keeping full vectors or not; saving and loading byte for byte (uneven parts, untrained); every truncation and flipped bit of a file, too many parts and a missing centroid behind valid checksums; with truncation; training final with codes only; out of memory. A golden file with PQ codes (`golden_hnsw_pq_v3.bin`) must be reproduced on every platform |
| `ivf` | 12 | Phase Q5: untrained scans everything (identical results); automatic and explicit training, its errors, and training again; probing every list exactly the exact results (every metric); recall rising with probes; every slot path keeping every vector in its nearest list (swap with last, updates moving lists, upserts, batches, a cancelled batch, `remove_where`, `compact`); IVF-PQ, re-ranked and codes only; selective filters and predicates still returning k results, deadlines; batch searches, and exact range searches; saving and loading byte for byte; every truncation and flipped bit of a file, and seven invalid `IVFL` contents; searching while another thread trains the lists; out of memory. A golden file with a trained inverted file (`golden_flat_ivf_v3.bin`); the model-based test also draws Flat indexes with an inverted file |
| `edges` | 11 | A review of edge cases across the SIMD kernels, PQ and IVF: one list (an exact scan), probes beyond the number of lists, k = 0; lists emptied and refilled; a filter nothing matches; identical vectors only; IVF with truncation, cosine zero vectors, `remove_batch`, `set_metadata_where`, and adding after loading; PQ with one dimension per part and trained on a single vector; all three codecs trained on values spanning more than the float range (automatic training included; shown to fail with the old float arithmetic); kernel remainders; SQ8 fixed answers unchanged; branches found untested by coverage (codec misuse, `list_of`, a trained codec section too short for its last field, trained lists with none configured) |
| `vamana` | 9 | Phase Q6a: recall of the graph searched in memory (L2, cosine); structure (degrees at most R, no self loops or repeated edges, every node reachable from the medoid); alpha 1.0 against 1.2 (recorded); determinism; one and two vectors, fewer vectors than R, identical vectors, one dimension; invalid input; search parameters; cosine normalization; a large group of identical vectors among distinct ones |
| `disk` | 34 | Phases Q6b to Q6d: every node read back equal to the graph built in memory; three block layouts; damaged files (header, sizes behind a valid checksum, truncation, trailing data, every section, a damaged node block met when read); invalid input; replacing a file only when complete; metadata and schemas; concurrent reads; recall against reads and rounds; exact distances; invalid searches; deadlines; concurrent searches; damage met during a search; alpha against reads; filtered searches (graph, direct, widening until k are found, shown to fail without widening), predicates, records, `search_by_id`, batches, concurrent filtered searches; files inconsistent behind valid checksums, each rejected by its own check; concurrent reads (identical results and statistics for every thread count and beam width; the overlap measured with simulated latency; an error during concurrent reads, then destroying the index; many searches sharing the readers); building from files (byte-identical to `build` when it fits; partitioned builds with every vector reachable and recall checked against an exact index; IDs, metadata and cosine; invalid files and settings, leaving an existing index and no temporary files; vectors files; the budget held with 3 replicas) |
| `stress` | 12 | Limits and long-run memory: a filter chained 200,000 times stops cleanly at the depth limit; a balanced 4,096-condition filter; the string dictionary bounded under 100,000 inserts of unique strings; dictionary compaction keeping filters and payload counts correct; thread counts capped; upper-level link blocks recycled over 30,000 churn steps; `compact()` releasing visited lists; unsigned values beyond `int64` rejected; a 10,000-value `in()` list; 60,000 mixed operations with bounded slots, dictionary and link memory; out-of-memory sweeps of dictionary compaction and block recycling |
| `concurrency` | 24 | Up to 8 threads searching HNSW and Flat at once, every answer matching the single-threaded one; plus scenarios C1 to C20 for the index lock: searches during adds, updates, saves and compaction (results always sorted, unique, finite, and every saved snapshot valid), several writers, the same ID added or removed by two threads (exactly one wins), atomic `upsert`, metadata and planner changes during filtered searches, predicates searching their own and other indexes while writers queue (no deadlock), changes from predicates refused on every thread, writers not starved, exceptions (including a predicate's own, on batch worker threads) and running out of memory releasing the lock, the nesting limit, accessors and saves during compaction, and simultaneous saves to one file (each complete, no temporary files left). Tests that could deadlock run under a watchdog that fails the run instead of hanging it |
| `e2e` | 6 | Add, remove and re-add lifecycles for every metric; 6,000 vectors at 48 dimensions; 5,000 random adds, removes and searches checked against Flat after every step; all metrics on the same data |

Tests share large indexes where possible: a 3,000-vector HNSW and Flat pair per metric is built the first time a test needs it, then reused, so the whole suite runs in about 15 seconds in Release.

**Running tests at will:**

```
test_comprehensive                      # run every test
test_comprehensive --list               # list all 745 test names
test_comprehensive --group hnsw         # run one group (repeatable)
test_comprehensive recall               # every test whose "group.name" contains "recall"
test_comprehensive flat.k_zero          # a single test
test_comprehensive --exclude e2e        # skip matching tests (repeatable)
test_comprehensive --fail-fast          # stop at the first failing test
test_comprehensive --help               # show all options
```

On Windows the program is `.\build\test_comprehensive.exe`. In CLion, put the same options in the `test_comprehensive` run configuration's **Program arguments** field.

**Output.** Each test prints PASS, FAIL or SKIP with its time. A failing check prints its line number and expression, and the test continues (`CHECK`) unless the check was essential (`REQUIRE`). The run ends with a summary listing every failed test, and the exit code is non-zero if anything failed:

```
hnsw-lite comprehensive tests | kernel: avx512 | 745 of 745 tests selected

[layer1]
  PASS  round_up_boundaries                                0.0 ms
  PASS  constants                                          0.0 ms
  ...
745 passed, 0 failed, 0 skipped, 0 not run, 36290 checks, 94.85 s
All selected tests passed.
```

**Writing a new test** takes one block; it is registered automatically:

```cpp
TEST(flat, my_new_case) {
    FlatIndex f(2, Metric::L2);
    f.add(1, std::vector<float>{0, 0});
    CHECK(f.size() == 1);
    CHECK_THROWS_AS(std::invalid_argument, f.add(1, std::vector<float>{1, 1}));
}
```

**How the out-of-memory tests work.** The test program replaces the global `operator new` and `operator delete` with versions that behave normally until told to fail the Nth allocation. Each test runs an operation with N = 0, then 1, then 2, and so on, until it completes without hitting the failure, so *every* allocation point is tried. After each failure it checks that nothing changed (or, for HNSW inserts, that the index is still consistent and searchable). AddressSanitizer and ThreadSanitizer install their own allocators, so under them these 30 tests report SKIP; define `HNSW_TEST_NO_ALLOC_HOOK` to turn the hook off manually.

### Code coverage

Coverage is measured with gcov and gcovr. On the library code:

| Measure | Covered |
|---|---|
| Lines | **100%** (1,750 of 1,750) |
| Branches | **98%** (1,636 of 1,670) |

Excluded from measurement, each marked in the source with a `GCOVR_EXCL` comment that states the reason:

- **CPU-dependent branches** in `dispatch.cpp` (what happens on CPUs without AVX2 or AVX-512). They can only run on such CPUs; the emulated Nehalem and Haswell runs cover them.
- **Untestable code:** the arena's `assert`, the 4-billion-vector limit, the visited-list pool's safety-net `catch` (unreachable since room is reserved in advance), a gcov artifact on a closing brace, and three purely defensive branches (unique entry lists, a never-empty result, self-links that cannot exist).

The NEON kernel file is excluded on x86 because it compiles to nothing there. Likewise, the CI coverage job excludes any SIMD kernel its runner's CPU cannot execute (GitHub's Linux runners usually lack AVX-512); those kernels are still tested on any machine that supports them. When the concurrency tests run, plain gcov counters race between threads and can move a branch count by one, which is why the branch figure can read 99% locally; the CI coverage job uses `-fprofile-update=atomic` for exact counts.

**Running it locally (Linux):**

```bash
cmake -B build-cov -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_CXX_FLAGS=--coverage -O0 -fno-inline -fprofile-update=atomic" -DCMAKE_EXE_LINKER_FLAGS=--coverage
cmake --build build-cov --target test_comprehensive
./build-cov/test_comprehensive
gcovr -r . build-cov --merge-lines --filter "$PWD/include/" --filter "$PWD/src/" --exclude "$PWD/src/distance_neon.cpp" --exclude-unreachable-branches --exclude-throw-branches --html-details coverage.html
```

**What the coverage audit found.** The first measurement showed 99% of lines and 94% of branches. Each gap was either tested, or excluded with a reason. Closing them added 10 tests and found one real bug (number 11 below): with graph repair turned off and nearly everything removed, a search could start in a region of the graph that reaches no live vector and return nothing at all.

**Second coverage audit (after the search features).** CI caught line coverage at 98.0%, and branch coverage turned out to be 92.9%. Three causes: template functions (gcov counts each instantiation separately, so executed lines were reported as missed; `--merge-lines` combines them), genuinely untested search-feature code (closed with 10 tests: every filter compilation error, number comparisons beyond 2^53 and 2^63, every field type through every metadata operation, exact counts for every filter kind, `plan_search` called directly, and out-of-memory failures in the thread pool), and two conditions that could never be true (simplified away). The remaining uncovered branches are compiler-generated (inside initializer lists and standard-library code), so the CI branch gate is 97%, one point below the measured value; the line gate stays at 99% with 100% measured. The coverage run takes several minutes, so it uses atomic counters (`-fprofile-update=atomic`) for exact counts with threads.

### CTest

CTest runs the comprehensive suite as one entry per group, 36 entries in total. Each group runs in its own process, so a crash in one group cannot stop the others:

```
 1/36 Test # 1: comprehensive.layer1 ................   Passed
 2/36 Test # 2: comprehensive.layer2 ................   Passed
 3/36 Test # 3: comprehensive.helpers ...............   Passed
 4/36 Test # 4: comprehensive.flat ..................   Passed
 5/36 Test # 5: comprehensive.hnsw ..................   Passed
 6/36 Test # 6: comprehensive.robustness ............   Passed
 7/36 Test # 7: comprehensive.deletion ..............   Passed
 8/36 Test # 8: comprehensive.concurrency ...........   Passed
 9/36 Test # 9: comprehensive.metadata ..............   Passed
10/36 Test #10: comprehensive.filter ................   Passed
11/36 Test #11: comprehensive.planner ...............   Passed
12/36 Test #12: comprehensive.payload ...............   Passed
13/36 Test #13: comprehensive.batch .................   Passed
14/36 Test #14: comprehensive.range .................   Passed
15/36 Test #15: comprehensive.search_e2e ............   Passed
16/36 Test #16: comprehensive.update ................   Passed
17/36 Test #17: comprehensive.hardening .............   Passed
18/36 Test #18: comprehensive.persistence ...........   Passed
19/36 Test #19: comprehensive.add_batch .............   Passed
20/36 Test #20: comprehensive.bulk ..................   Passed
21/36 Test #21: comprehensive.records ...............   Passed
22/36 Test #22: comprehensive.control ...............   Passed
23/36 Test #23: comprehensive.codec .................   Passed
24/36 Test #24: comprehensive.sq8 ...................   Passed
25/36 Test #25: comprehensive.matryoshka ............   Passed
26/36 Test #26: comprehensive.lean ..................   Passed
27/36 Test #27: comprehensive.review ................   Passed
28/36 Test #28: comprehensive.codes_only ............   Passed
29/36 Test #29: comprehensive.binary ................   Passed
30/36 Test #30: comprehensive.pq ....................   Passed
31/36 Test #31: comprehensive.ivf ...................   Passed
32/36 Test #32: comprehensive.edges .................   Passed
33/36 Test #33: comprehensive.vamana ................   Passed
34/36 Test #34: comprehensive.disk ..................   Passed
35/36 Test #35: comprehensive.stress ................   Passed
36/36 Test #36: comprehensive.e2e ...................   Passed
100% tests passed, 0 tests failed out of 36
```

Run one group through CTest with, for example, `ctest --test-dir build -R comprehensive.hnsw`.

In a Debug build the comprehensive suite takes about a minute, and several times longer with sanitizers, because it builds many indexes with optimizations off.

### Verification

- **CPUs:** all tests pass on every CPU type below. Each kernel rounds slightly differently, so each CPU builds a slightly different graph; passing on all of them shows the HNSW tests do not depend on one exact graph.

  | CPU | Kernel versions tested | Version chosen |
    |---|---|---|
  | Modern x86-64 with AVX-512 | scalar, AVX2, AVX-512 | AVX-512 |
  | Emulated Intel Haswell (2013) | scalar, AVX2 | AVX2 |
  | Emulated Intel Nehalem (2008) | scalar | scalar |
  | Emulated 64-bit ARM | scalar, NEON | NEON |

- **Compilers:** all tests pass with GCC 13 and Clang 18. MSVC, MinGW on Windows and Apple Clang are covered by the [CI pipeline](#continuous-integration), which also runs the suite on real ARM hardware.
- **Sanitizers:** all tests pass with AddressSanitizer and UndefinedBehaviorSanitizer (the out-of-memory tests are skipped there), and the threaded groups pass under ThreadSanitizer with no data races reported.
- **Warnings:** none with `-Wall -Wextra -Wpedantic`, nor with `-Wconversion -Wshadow` (which approximate MSVC's `/W4` conversion warnings), under both GCC and Clang.
- **Comparison with hnswlib:** where results looked low, the same data and operations were run through the reference hnswlib library. Inner-product recall on the lifecycle data is 0.898 / 0.905 / 0.860 (add / remove / re-add) against hnswlib's 0.887 / 0.900 / 0.853, confirming that lower inner-product numbers come from the metric, not this implementation.

### Bugs found by testing, and fixed

1. `Storage::insert` with an invalid level stored the ID and vector but not the graph node, leaving the three parts out of sync. Levels are now validated first.
2. `normalize` on very small vectors produced infinity and NaN, because the scale factor overflowed a float. It now scales in double.
3. NaN and infinity were accepted into indexes, silently corrupting distances. They are now rejected.
4. A very large `k` crashed searches by reserving memory for `k` results. `k` and `ef` are now capped.
5. Many identical vectors made unrelated vectors unreachable. The neighbor heuristic now skips exact copies.
6. HNSW inserts chose removed nodes as neighbors, wasting link slots: after removals and re-adds, inner-product recall was 0.792 versus hnswlib's 0.853 on identical data. Removed nodes are now skipped when choosing neighbors, giving 0.860.
7. `VectorStore` always allocated blocks of 65,536 rows, so storing a single 1,536-dimension vector allocated and zeroed 400 MB (a test took 609 ms). Blocks are now capped at 8 MB, and the test takes 4.6 ms.
8. **Running out of memory corrupted indexes.** `IdMap::add` registered the user ID before growing its arrays, so a failure left a stale ID that broke every later insert; `Storage::insert` had no rollback if a later step failed; and a half-linked HNSW vector stayed visible after a failure. Inserts are now all-or-nothing in `IdMap`, `VectorStore`, `GraphStorage`, `Storage` and `FlatIndex`, and a failed HNSW insert hides its half-linked vector. Run against the old code, 4 of the out-of-memory tests fail.
9. Huge finite values could make an inner product overflow to NaN (+∞ plus −∞), which breaks the ordering every heap and sort relies on. NaN distances now count as +∞, so such pairs sort last.
10. **A visited list could be silently dropped.** Returning a borrowed list to the pool used `push_back`, which can allocate; a failure there was swallowed (as a destructor path must) and the list discarded. Room is now reserved when each list is created, so returning one never allocates. The out-of-memory harness also became strict: a failure that an operation swallows is now reported as a test failure.
11. **A search could return nothing while live vectors existed** (found by the coverage audit). Links are one-directional, so after many removals without graph repair, the greedy descent could end on a removed node whose links lead only to other removed nodes; the search then found no live vector. Searches that find fewer than k results, and inserts that find no live neighbor, now retry with the entry point (always live) as an extra starting point. The insert retry replaces the old fallback of linking new vectors to removed nodes.
12. **A failed HNSW insert could leave a new metadata field behind** (found by the search-feature tests). If memory ran out while linking a vector, after its metadata had been written, the failure handler cleared the row but kept any dynamic field or dictionary string that insert had created. It now undoes the metadata write completely.

**Found by the load audit** (probing behavior under deep inputs, long churn and extreme settings), all fixed and covered by the `stress` group:

13. **A deeply nested filter crashed the program.** Chaining `&&` 200,000 times overflowed the stack. Filters are now limited to 256 levels; going deeper throws `std::invalid_argument` with a clear message.
14. **The string dictionary never shrank.** With 10 live vectors after 100,000 inserts of unique strings, it held 100,000 strings. Strings now carry reference counts, and unused ones are reclaimed automatically (and by `compact()`, now also on Flat); the same run keeps 672.
15. **HNSW slot reuse leaked upper-level link memory.** Allocations kept growing under churn (13,264 slots after 200,000 steps, with about 700 in use). Blocks are now recycled whenever a node's level changes; allocations plateau at 1,068.
16. **A huge thread count failed.** Asking batch search for 100,000 threads threw. Thread counts are now capped at max(64, 4 x hardware threads).
17. **Unsigned values above the `int64` range wrapped negative** in metadata and filters. They are now rejected.
18. **`compact()` kept visited lists sized for the old index.** They are now released.
19. **`in()` on keywords checked its values one by one** per vector; they are now sorted and binary-searched.
20. **An HNSW update could leave the new version with no links at all** (found by the update edge-case review). An update leaves the old version out of the new one's neighbors; in sparse, mostly removed graphs without repair, the old version could be the only live vector the search reached, so the new version linked to nothing and could never be found. It now links through the old version as a last resort: once retired, the old version stays traversable and bridges to the new one.

**Found by the hardening review** (Phase 1 of the [hardening and performance plan](hardening-and-performance-plan.md)), all fixed and covered by the `hardening` group:

21. **A batch search started from a predicate, inside a batch on the same index, hung forever.** The pool's threads all waited for each other. Nested batches now run inline on the calling thread.
22. **Changing an index from inside a predicate went undetected**, with undefined results. Indexes now refuse changes during a search with `std::logic_error`.
23. **Cosine distances could be slightly negative**, so radius-0 range search missed exact copies. Indexes now clamp cosine distances at 0.
24. **A huge `M` attempted a huge allocation.** `M` is now limited to 2048.
25. **Filters listing thousands of tags were slow** (136 ms over 20,000 vectors). Tags are now sorted and binary-searched (4.7 ms).

**Found while building saving and loading:**

26. **A save could write a damaged file without reporting an error when memory ran out.** C++ streams catch an allocation failure inside their buffer and only set an error flag; the random generator's state, written through a string stream, could be silently cut short. Saving and loading now switch every stream they use to rethrow such errors (`StreamErrorsThrow`). Found by the out-of-memory sweep of saving.
27. **Saved HNSW indexes were not portable between standard libraries.** The random generator's state was written through C++ streams, whose text form of `std::mt19937_64` differs between libstdc++ (Linux, MinGW) and libc++ (macOS): a file saved on one could not be loaded on the other. The golden files caught it on the macOS CI job before release. Random levels now come from SplitMix64, whose whole state is one 64-bit number saved as plain bytes; the fix was verified by building and testing against libc++. The same CI run showed that libc++ reports running out of memory inside a file stream as `std::ios_base::failure` rather than `std::bad_alloc`; the save still left the old file intact, and the test now accepts either.

**Found while building thread safety:**

28. **A deadlock in the first design of the index lock.** Thread 1, inside a search of index A, had a predicate that changed index B; thread 2, inside a search of B, had a predicate that searched A; a writer was queued on A. With writers preferred, each waited for the next. Found by test `c18` under its watchdog. Fixed with two rules: no change from inside any index operation, and nested reads skip the writer queue (see [Thread safety](architecture.md#thread-safety)).
29. **Simultaneous saves to one file could corrupt it.** Once saves could run in parallel, two saves to the same path wrote into one shared temporary file (`<path>.tmp`), overwriting each other mid-write, and a rename failed once another save had moved the file away. Found by reviewing the edge cases of the new lock. Every save now writes its own uniquely named temporary file. Test `c20` was confirmed to fail against the old naming (3 runs out of 3) and to pass with the fix.

**Found while building parallel inserts:**

30. **Parallel linking let threads reach half-linked vectors.** Vectors were linked level by level, top-down; once linked on an upper level, other threads could reach a vector and descend to a level it had not linked yet, finding an empty list, and links they added to it could later be overwritten. Found under ThreadSanitizer, whose interleaving made it frequent (graph quality checks in `b07` and `b08` failed). Fixed by publishing a vector only when complete: all its levels are decided, then its own lists set, and only then the links that make it reachable.
31. **A failed batch left links to its removed vectors.** Rolling back removed the batch's vectors without repairing the graph, so after many failures more than 2% of links pointed to removed vectors. Found by the out-of-memory sweep in `b11`. Rollback now repairs around them, best effort, as `remove` does.

**Found on CI's multi-core machines:**

32. **Two concurrency tests had flawed checks; the library was right in both.** `c08` compared two accessor calls (`deleted_count()` and `capacity()`) that a compaction on another thread could separate; it now checks invariants on one consistent snapshot (a save). `c14` demanded 100% graph reachability after thousands of upserts and used vectors shared between IDs. Running its upserts single-threaded in 200 random orders showed the cause was not threads but **update churn**, which weakens the graph whatever the order (see the [project notes](project-notes.md#limitations)); the test now uses unique vectors, accepts the measured degradation, and requires full reachability after `compact()`. A single-core machine rarely switches threads between two calls, which is why both only showed on CI.

**Found while building record reading:**

33. **The user guide overstated what a failed HNSW `add` restores.** It promised the index "exactly as it was"; the contents are (every vector, its metadata, search results), but if memory runs out while a vector is linked, its slot is kept free for the next add and its random level is used up, so the saved bytes differ. Found when a new test compared bytes after failed adds; an experiment showed the same for L2 and cosine, so it predates this work. The guide now states the guarantee precisely.

**Found while building deadlines and cancellation:**

34. **A progress callback could report failure for completed work.** The final progress report of `compact` was first made after its commit, so a throwing callback would have signalled failure for a compaction that had happened; likewise after `add_batch`'s rollback scope. Found by reviewing the code before testing. The final report now comes while the work can still be undone (test `t11` throws from it).

**Found while building training and codecs:**

35. **Floating-point contraction would have made ARM builds differ.** GCC fuses `a * b + c` into one instruction by default on ARM, changing rounding: built for ARM64 without `-ffp-contract=off`, SQ8 distances differed in their last bits from x86. Found by fingerprinting exact results on x86, emulated ARM64 with the flag, and without it. `kmeans.cpp` and `codec.cpp` are built with the flag, and fixed-answer tests run on every CI platform.

**Found while building `keep_full_vectors`:**

36. **A saved setting was lost on loading.** `CONF` restores `keep_full_vectors`, but reading the later `CODC` section replaced all quantization settings with its own, resetting the flag to its default: a lean index would have reloaded keeping full vectors (the same had been handled for `truncate_to`, but not for the new flag). Found by tests `l04` and `l07` (save, load, compare) before release; loading now keeps both settings from `CONF`.

**Found by reviewing phases Q0 to Q2 and `keep_full_vectors`:**

37. **A range search nested in a coded search used the wrong query.** While a search compares codes, a per-thread marker tells `distance_to` which query table to use; range searches installed none, so one started from a coded search's predicate compared codes with the outer search's query. Found by reading the code; the first test missed it (it ran the nested search only on the predicate's first call, which comes from the planner's sampling, before the search), the corrected test (`v01`, every call) failed. Range searches now install a full-precision marker.
38. **`k * rerank_factor` could wrap around.** With `rerank_factor = 2^63`, an exact search for 10 results asked for 0 candidates and returned nothing. Found by reading the code, proven by `v02`; the product now saturates.
39. **`compact()` slightly changed stored vectors without originals.** It rebuilt from the stored vectors, normalizing already normalized cosine vectors again, which can change their last bits (a distance differed in its 7th digit). Found by the model-based test `v03` (exact HNSW against Flat); compaction now copies stored vectors exactly, and `v04` (shown to fail with the old code) checks they stay bit for bit the same over two compactions.

**Found while building SQ8 with codes only:**

40. **A float read hidden behind a local alias.** `same_vector` (duplicate detection during graph building) read floats through a local `VectorStore&`, which a search for `vectors().get` missed. Caught at once by the safety net (reading a dropped value throws), located with a debugger; it now compares codes.
41. **Retraining a codes-only index could collapse it for good.** The model-based test found indexes whose 210 and 249 vectors all shared a single code: retraining on a few live vectors gave a range of zero width, and every later training, learning only from decoded vectors, kept it. Training again is now refused once only codes are kept; the test expects the refusal.

**Found while building the SIMD kernels and PQ:**

42. **Binary distances were about 8 times slower than full-precision ones.** The plain version tested bits one at a time (4.2 microseconds per 768-dimension distance on a desktop PC, against 0.47 for floats), so binary codes saved memory at a real cost in speed. Found by the new `bench_codecs`; the SIMD kernels made them faster than float distances.
43. **A damaged-file test would have stopped testing its rule.** Adding the number of PQ parts at the end of `CODC` made `c05`'s deliberately inconsistent file too short, so it would have been rejected for that instead of the rule it exists for; the test now appends the field.

**Found while building IVF:**

44. **Training could have left a trained index half-updated.** The first version put the new centroids into the index and resized its live list table to reuse the search code, before assigning vectors: running out of memory there would have broken the all-or-nothing promise and could have truncated existing lists. Found on re-reading the code before any test ran; training now builds everything in local variables and swaps it in.
45. **`<cmath>` was included only by accident** in `persistence.cpp` (through another header), a likely build failure with other standard libraries such as MSVC's; it is now included explicitly.

**Found by the edge-case review after IVF:**

46. **SQ8 could not train on values spanning more than the float range.** Its step, `(max - min) / 255`, was computed in float, so `max - min` overflowed to infinity for values from -3e38 to 3e38: explicit training threw, and automatic training (which retries after running out of memory) failed silently, leaving the index untrained for good. Encoding had the same overflow (`value - min`), which gave such values the top code. Found by a probe of boundary conditions; training, encoding and decoding now compute in double (decoded values kept within the float range), with every ordinary code unchanged (the fixed answers and golden files are the same, on x86 and ARM64). Test `e06` fails with the old arithmetic.
47. **A check that can never fail.** Coverage showed that loading's check for a non-finite IVF centroid is never reached; printing the errors showed why: the file reader already rejects every non-finite float, so `i10`'s crafted NaN centroid is caught there. The check is kept as a second line of defense and marked as unreachable for coverage. Every other crafted edit in `i10` was confirmed to be rejected by its own rule.

**Found while building the disk-based index:**

48. **Vamana pruning removed edges from complete graphs.** With fewer vectors than R the starting graph is complete, but the passes still pruned it, leaving vectors unreachable. Found by `va05`; tiny inputs now keep the complete graph.
49. **Identical vectors collapsed the graph.** At distance 0 the pruning rule drops every other candidate, and ties broken by node number made the lowest-numbered copies hubs while the others became unreachable (607 of 660 reachable in `va09`). Identical candidates are no longer pruned, and ties are ordered by a hash of the node pair.
50. **Vamana does not guarantee every node an incoming edge.** Even after both fixes one of 100 identical vectors was never chosen as a neighbor; a first repair pass then ping-ponged inside a large group of copies (656 of 660). A connectivity repair now prefers free slots and neighbors keeping another incoming edge: every node is reachable in every test.
51. **Coarse navigation codes wasted reads.** Disk recall started well below the graph's own (0.735 against 0.99): with one PQ part per 8 dimensions the codes misjudged which nodes to read. Measured across resolutions, finer codes restored recall without adding reads; the disk index now defaults to one part per 2 dimensions.
52. **Checks behind checksums were unreached.** Coverage showed that `open`'s checks of section contents (PQ, IDs, metadata) and of node records were never run: damaged-file tests are caught by the checksums first. `dk24` re-signs every checksum after its edits and confirms each check by its message.

**Found while extending the disk-based index:**

53. **The memory budget was exceeded by 1.6 times.** k-means makes partitions of unequal size, so a dense region's partition held 2,692 vectors against a cap of 1,706. Full partitions are now skipped for the next nearest.
54. **With 3 replicas the budget broke again** (1,347 against 853): the automatic partition count assumed at most 2 replicas, and a fallback for "every partition full" overfilled one silently. The count is now provably sufficient (at least ceil(n * replicas / capacity) + replicas), explicit counts below it are rejected, and the fallback is gone. Test `dk34`.
55. **Skipping full partitions cost recall** (0.952 instead of 1.000): vectors pushed out of their nearest partition lose their natural neighbors. Measured across partition counts; the automatic count now allows two thirds of slack, which restored 1.000.
56. **`near` is a macro on Windows.** `<windows.h>` defines `near` and `far` (from 16-bit Windows), so a variable named `near` broke the MinGW build while GCC and Clang on Linux compiled it; found by the MinGW check before delivery and renamed.

The model-based test also showed two expected behaviors, now documented: results with equal distances may come in a different order in HNSW and Flat (and after `compact()`), and inner-product indexes whose vectors differ in length by orders of magnitude leave some vectors hard to reach.

A test also looped over `*f.search_by_id(...)` directly and read freed memory, failing at random (about 60% of runs): a range-based `for` destroys the temporary optional before the loop body. The test stores the result first now, and the [user guide](user-guide.md#searching-by-stored-id) warns about this trap.

Two test assumptions were also wrong and corrected: a zero vector is valid with cosine distance (as documented), and graphs with `M = 2` are poorly connected however they are built (about 89% of vectors reachable even one by one), so `b08` compares with one-by-one adds.

## Continuous integration

Every change is built and tested automatically by GitHub Actions ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

**When it runs:**

- On every **pull request into `master`**, and again on every new push to that pull request, so changes are checked before they are merged.
- On every **push to `master`**, to confirm the merged result.
- **Manually**, from the Actions tab with "Run workflow".

It does not run on pushes to a branch without a pull request; that would run everything twice once a pull request exists. To get feedback early, open a **draft pull request** as soon as the branch is created. A newer run for the same branch cancels an older one that is still running.

**Documentation-only changes skip the builds.** A first job, *Detect changes*, lists the files a pull request (or push) changes, using [classify_changes.sh](../.github/scripts/classify_changes.sh). Documentation is `docs/`, any `.md` file and `LICENSE`; everything else is code, including the workflows themselves. The builds and tests below run only when code changed; the *Documentation links* job runs only when documentation changed, checking every relative link and anchor in every Markdown file with [check_links.py](../.github/scripts/check_links.py) (run it locally from the repository root the same way). Skipped jobs count as passed. Manual runs, the first push of a branch, and any case where the base commit is unknown run everything.

**The one check to require** in branch protection (Settings, Branches, the rule for `master`, "Require status checks to pass") is **CI passed**: it passes when every job that ran succeeded, and fails if any job failed or was cancelled, including the change detection itself.

**What it runs when code changed (14 jobs in parallel):**

| Job | Build | What it adds |
|---|---|---|
| Linux / GCC | Release | Full test suite, plus both benchmarks as a smoke test |
| Linux / Clang | Release | Full test suite with a second compiler |
| macOS ARM / Apple Clang | Release | Full test suite on **real ARM hardware** (Apple Silicon), using the NEON kernels |
| Windows / MSVC | Release | Full test suite with Microsoft's compiler, including the MSVC-specific CPU detection |
| Windows / MinGW GCC | Release | Full test suite with the MinGW toolchain, including the out-of-memory tests on Windows |
| Linux / ASan + UBSan | Debug | Memory errors, leaks and undefined behavior (out-of-memory tests report SKIP) |
| Linux ARM64 / GCC (NEON) | Release | Full suite on ARM with GCC and libstdc++ (macOS covers ARM only with Apple's compiler and libc++) |
| Linux / Clang + libc++ | Release | Full suite against libc++, macOS's standard library; catches library differences quickly, like the saved-file portability bug |
| Linux / GCC 11 and Linux / Clang 14 | Release | The oldest compilers the README promises to support (Ubuntu 22.04) |
| Linux / emulated Nehalem and Haswell | Release | The plain C++ and AVX2 distance code on emulated CPUs (QEMU), which GitHub's machines (AVX2 or better) never exercise; skips the two exhaustive file-corruption sweeps and the stress group, which are slow under emulation |
| Linux / ThreadSanitizer | RelWithDebInfo | Data races in the concurrency, helpers, deletion, add_batch, codec, sq8, matryoshka, lean, review, codes_only, binary pq, ivf and disk groups |
| Linux / Coverage | Debug | Line and branch coverage; fails if it drops below 99% of lines or 97% of branches, and uploads an HTML report |

Each job fails on its own, so one broken platform never hides the results of the others.

**Protecting `master`.** To make passing CI a requirement for merging: on GitHub, open **Settings → Branches → Add branch protection rule** for `master`, enable **Require status checks to pass before merging**, and select the CI jobs. Together with **Require a pull request before merging**, nothing reaches `master` without passing every platform.

### Nightly run

A second workflow, [nightly.yml](../.github/workflows/nightly.yml), runs every night at 02:00 UTC on `master`, and on demand from the Actions tab (**Nightly**, then **Run workflow**). It runs what pull requests skip to stay fast:

| Job | What it adds to the pull-request checks |
|---|---|
| Emulated Nehalem and Haswell, full suite | The two exhaustive file-corruption sweeps and the stress group, which pull requests skip under emulation |
| ASan + UBSan, full suite | The same as on pull requests, but checking `master` itself every night |
| ThreadSanitizer, full suite | Every test group, where pull requests check only the threaded groups |

**The report:** each run's page shows a table of job results, and for each job the distance code used, the totals, any failed tests and the first lines of any sanitizer report (made by [summarize_tests.sh](../.github/scripts/summarize_tests.sh)). Each job's full test log is attached as an artifact for 14 days. If any job fails, an issue labelled `nightly-failure` is opened, or, if one is already open, commented on, so a recurring failure produces one issue rather than one per night. Watching the repository on GitHub turns these into email notifications.

GitHub pauses scheduled workflows on public repositories after 60 days without activity (and emails a warning first); any commit restarts them.