# Testing and continuous integration

How hnsw-lite is tested: the test suite and its runner, code coverage, the CI pipeline, and the bugs testing has found.

## Testing

### The comprehensive suite

`tests/test_comprehensive.cpp` contains **every test scenario in one program: 418 individually named tests** in 17 groups, with a built-in runner. It needs no external test framework.

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
| `stress` | 12 | Limits and long-run memory: a filter chained 200,000 times stops cleanly at the depth limit; a balanced 4,096-condition filter; the string dictionary bounded under 100,000 inserts of unique strings; dictionary compaction keeping filters and payload counts correct; thread counts capped; upper-level link blocks recycled over 30,000 churn steps; `compact()` releasing visited lists; unsigned values beyond `int64` rejected; a 10,000-value `in()` list; 60,000 mixed operations with bounded slots, dictionary and link memory; out-of-memory sweeps of dictionary compaction and block recycling |
| `concurrency` | 4 | Up to 8 threads searching HNSW and Flat at once, mixing indexes, metrics and ef values; every answer must match the single-threaded one |
| `e2e` | 6 | Add, remove and re-add lifecycles for every metric; 6,000 vectors at 48 dimensions; 5,000 random adds, removes and searches checked against Flat after every step; all metrics on the same data |

Tests share large indexes where possible: a 3,000-vector HNSW and Flat pair per metric is built the first time a test needs it, then reused, so the whole suite runs in about 15 seconds in Release.

**Running tests at will:**

```
test_comprehensive                      # run every test
test_comprehensive --list               # list all 418 test names
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
hnsw-lite comprehensive tests | kernel: avx512 | 418 of 418 tests selected

[layer1]
  PASS  round_up_boundaries                                0.0 ms
  PASS  constants                                          0.0 ms
  ...
418 passed, 0 failed, 0 skipped, 0 not run, 26343 checks, 16.01 s
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

**How the out-of-memory tests work.** The test program replaces the global `operator new` and `operator delete` with versions that behave normally until told to fail the Nth allocation. Each test runs an operation with N = 0, then 1, then 2, and so on, until it completes without hitting the failure, so *every* allocation point is tried. After each failure it checks that nothing changed (or, for HNSW inserts, that the index is still consistent and searchable). AddressSanitizer and ThreadSanitizer install their own allocators, so under them these 27 tests report SKIP; define `HNSW_TEST_NO_ALLOC_HOOK` to turn the hook off manually.

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

CTest runs the comprehensive suite as one entry per group, 17 entries in total. Each group runs in its own process, so a crash in one group cannot stop the others:

```
 1/17 Test # 1: comprehensive.layer1 ................   Passed
 2/17 Test # 2: comprehensive.layer2 ................   Passed
 3/17 Test # 3: comprehensive.helpers ...............   Passed
 4/17 Test # 4: comprehensive.flat ..................   Passed
 5/17 Test # 5: comprehensive.hnsw ..................   Passed
 6/17 Test # 6: comprehensive.robustness ............   Passed
 7/17 Test # 7: comprehensive.deletion ..............   Passed
 8/17 Test # 8: comprehensive.concurrency ...........   Passed
 9/17 Test # 9: comprehensive.metadata ..............   Passed
10/17 Test #10: comprehensive.filter ................   Passed
11/17 Test #11: comprehensive.planner ...............   Passed
12/17 Test #12: comprehensive.payload ...............   Passed
13/17 Test #13: comprehensive.batch .................   Passed
14/17 Test #14: comprehensive.range .................   Passed
15/17 Test #15: comprehensive.search_e2e ............   Passed
16/17 Test #16: comprehensive.stress ................   Passed
17/17 Test #17: comprehensive.e2e ...................   Passed
100% tests passed, 0 tests failed out of 17
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

## Continuous integration

Every change is built and tested automatically by GitHub Actions ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

**When it runs:**

- On every **pull request into `master`**, and again on every new push to that pull request, so changes are checked before they are merged.
- On every **push to `master`**, to confirm the merged result.
- **Manually**, from the Actions tab with "Run workflow".

It does not run on pushes to a branch without a pull request; that would run everything twice once a pull request exists. To get feedback early, open a **draft pull request** as soon as the branch is created. A newer run for the same branch cancels an older one that is still running.

**What it runs (8 jobs in parallel):**

| Job | Build | What it adds |
|---|---|---|
| Linux / GCC | Release | Full test suite, plus both benchmarks as a smoke test |
| Linux / Clang | Release | Full test suite with a second compiler |
| macOS ARM / Apple Clang | Release | Full test suite on **real ARM hardware** (Apple Silicon), using the NEON kernels |
| Windows / MSVC | Release | Full test suite with Microsoft's compiler, including the MSVC-specific CPU detection |
| Windows / MinGW GCC | Release | Full test suite with the MinGW toolchain, including the out-of-memory tests on Windows |
| Linux / ASan + UBSan | Debug | Memory errors, leaks and undefined behavior (out-of-memory tests report SKIP) |
| Linux / ThreadSanitizer | RelWithDebInfo | Data races in the concurrency, helpers and deletion groups |
| Linux / Coverage | Debug | Line and branch coverage; fails if it drops below 99% of lines or 97% of branches, and uploads an HTML report |

Each job fails on its own, so one broken platform never hides the results of the others.

**Protecting `master`.** To make passing CI a requirement for merging: on GitHub, open **Settings → Branches → Add branch protection rule** for `master`, enable **Require status checks to pass before merging**, and select the CI jobs. Together with **Require a pull request before merging**, nothing reaches `master` without passing every platform.