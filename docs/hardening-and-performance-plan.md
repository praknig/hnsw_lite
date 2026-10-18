# Hardening and performance plan

This plan combines two pieces of work that depend on each other: closing the edge cases a design review found, and making search and build faster. It sets out what will be done, in which order, and how each step is proven. Nothing here is implemented yet.

Related documents: the [core capabilities plan](core-capabilities-plan.md) (persistence and concurrency, which this plan schedules), [Performance and tuning](performance.md) (current benchmarks) and [Testing](testing.md).

## Contents

- [Why one plan](#why-one-plan)
- [Starting point](#starting-point)
- [Part 1: Edge cases](#part-1-edge-cases)
- [Part 2: Optimizations](#part-2-optimizations)
- [Part 3: Order of work](#part-3-order-of-work)
- [Rules for every phase](#rules-for-every-phase)
- [Decisions needed](#decisions-needed)
- [Status](#status)

## Why one plan

The two pieces of work share risks and constrain each other's order:

- **Hardening protects optimization.** The edge-case tests come first, so they form the safety net for changes to the hottest code.
- **Optimizations create new edge cases.** Allocation-free search, for example, would break predicates that search the same index, which works today. Each optimization's risks are listed next to it, so they are tested as part of the same work.
- **The memory layout must be settled before persistence.** Saving and loading fixes the file format. Changing the layout afterwards (storing links next to vectors, or 8-bit quantization) would immediately need a second format version and a conversion path.

## Starting point

Measured during the review: 50,000 synthetic vectors of 128 dimensions, AVX-512, one thread, Release build.

| Measure | Value |
|---|---|
| HNSW search, `ef = 80` | About 14,000 queries per second at 99.8% recall |
| Flat search | About 850 queries per second |
| HNSW build (`M = 16`, `ef_construction = 200`) | 6.9 seconds, single-threaded |

A function-level profile of search and build together (gprof, with inlining disabled, so the percentages are approximate):

| Where the time goes | Share |
|---|---|
| Distance computations (about 105 million calls; the kernel is already SIMD) | About 45% |
| `search_level`: the candidate heap, visited-list checks, and waiting on memory as each step jumps to a new node | About 40% |
| Everything else (neighbor selection, link updates, duplicate checks) | About 15% |

**Conclusion:** the remaining cost is memory traffic and bookkeeping, not arithmetic. This data set fits largely in the CPU cache, which hides the memory effects that dominate at scale. That is why Phase 0 measures on real, larger data sets before any optimization is chosen.

## Part 1: Edge cases

Found by reviewing the design; none has a test yet. Each becomes at least one test scenario (IDs H1 onwards in the test plan written during Phase 1).

### Likely problems

| ID | Edge case | Risk | Planned response |
|---|---|---|---|
| E1 | "Same seed gives an identical graph" may hold only on one platform: if the random level draw uses `std::uniform_real_distribution`, GCC, Clang and MSVC produce different numbers | The documentation promises more than is true; saved files would differ by platform | Check the level draw; if affected, derive levels from the generator's raw bits (portable), or qualify the promise |
| E2 | A predicate that calls back into the index | A predicate calling `search_batch` on the same index during a batch deadlocks on the shared pool; one that changes the index is undefined behavior. Neither is documented or detected | Decision D2: support re-entrant searches, or document and detect calls from inside a predicate |
| E3 | Range search with cosine at radius 0 | Rounding can give identical normalized vectors a distance of −0.0000001 or +0.0000001, so exact copies may be included or missed unpredictably | Clamp cosine distances at 0, or document a tolerance |
| E4 | No upper limits on HNSW settings | A huge `M`, or `ef_construction` far below `M`, may be accepted, giving huge memory use or poor graphs | Add upper bounds with clear errors |

### Unverified: a test decides

| ID | Edge case |
|---|---|
| E5 | User IDs at the extremes (0 and the maximum 64-bit value) through add, update, remove, filters and batch search |
| E6 | `update` given a vector that points into the index's own storage |
| E7 | Constructing an index with dimension 0 |
| E8 | Thousands of fields in a dynamic schema: memory and speed per row |
| E9 | `has_any_tag` and `has_all_tags` with thousands of tags (checked one by one, unlike `in`, which is sorted) |
| E10 | A predicate that throws during a multi-threaded batch, tested at the index level |
| E11 | Peak memory of batch search: each thread holds a visited list sized to the whole index (64 threads at 1 million vectors is 256 MB) |
| E12 | Subnormal (very small) floats in vectors, which slow some CPUs dramatically |
| E13 | The filter depth limit on Windows pool threads (1 MB stacks instead of Linux's 8 MB) |

## Part 2: Optimizations

Expected gains are estimates, based on what similar changes achieve in hnswlib and Faiss. Phase 0 measurements decide which are pursued. Each row also lists the edge cases the change itself creates, which are tested as part of it.

### Search

| ID | Optimization | Expected gain | Effort | Edge cases it creates |
|---|---|---|---|---|
| O1 | **Prefetching:** start loading the next neighbors' vectors while the current distance is computed | 10–30% on indexes larger than the cache | Small | Never prefetch an empty link slot or an out-of-range position |
| O2 | **Batch neighbor distances:** collect a node's unvisited neighbors, then compute their distances in one tight loop | 5–15% | Small | Results must be identical to today's |
| O3 | **No allocations per search:** reuse per-thread scratch space for the query copy, heap and results | 3–8%, more for small `k` | Small | Re-entrancy: a predicate that searches on the same thread must not overwrite the outer search's scratch space (see E2) |
| O4 | **16-bit visited marks** instead of 32-bit | A few % on large indexes | Small | Wraparound after 65,535 searches: the marks must be cleared then |
| O5 | **Store each node's links next to its vector:** one memory region per graph step instead of two | 10–25% | Large (Layer 1 redesign) | Slot reuse, `compact()`, alignment and upper-level links all change |
| O6 | **8-bit scalar quantization (SQ8):** 4 times less memory and bandwidth, with an optional exact re-check of the top results | 2–3 times faster, 4 times less memory, slight recall cost | Large (new feature) | Dimensions with a zero value range; values outside the trained range; cosine; updates and removals |

### Build

| ID | Optimization | Expected gain | Effort | Edge cases it creates |
|---|---|---|---|---|
| O7 | **Parallel build** (concurrency stage B, with `add_batch`) | Close to the core count | Large | The graph depends on thread timing: a single-threaded mode is needed for reproducible builds and tests |
| O8 | **Cheaper duplicate check:** compare whole vectors only when the distance is exactly 0 (24 million comparisons per 50,000-vector build today) | 1–3% of build | Tiny | None beyond the existing duplicate tests |
| O9 | **Remember neighbor distances** instead of recomputing them when re-pruning a full list | 10–20% of build | Medium | Extra memory per link; stored distances must stay consistent through updates, removals and repair |

### Filtered and Flat search

| ID | Optimization | Expected gain | Effort | Edge cases it creates |
|---|---|---|---|---|
| O10 | **Payload posting lists:** exact filtered search reads only the matching vectors | Large for selective filters on big indexes | Medium | All-or-nothing updates under out-of-memory; memory under churn |
| O11 | **One Flat query split across threads** | Close to the core count | Small | Tie order must stay identical when merging per-thread results |
| O12 | **Matrix-multiply kernel for Flat batches** | 2–4 times for large batches | Medium | Different summation order gives slightly different distances, breaking "batch results identical to separate searches" (decision D3) |

### Found after Phase 1 (N1 to N10)

A second review of the plans and limitations found these:

| ID | Item | Kind | Phase |
|---|---|---|---|
| N1 | Skip bounds checks inside the search loop, where positions are already known valid (keep them in debug builds and the public API) | Optimization | 2 |
| N2 | Reusable compiled filters, instead of recompiling on every search | Optimization | 7 |
| N3 | Cached selectivity estimates for repeated filters, until the index changes | Optimization | 7 |
| N4 | Let the user choose the distance code (for example AVX2 where AVX-512 lowers the clock speed) | Option | 2 |
| N5 | Optional automatic `compact()` when removed slots pass a set fraction | Option | 4 |
| N6 | Each concurrent search holds a visited list sized to the index: 64 threads at 1 million vectors use about 256 MB | Edge case | Next hardening pass |
| N7 | Alternating thread counts in `search_batch` recreate the thread pool each time | Edge case | Next hardening pass |
| N8 | Saving needs offsets instead of the arena's raw pointers; plan for versions, corrupted files and safe writes | Edge case | 5 |
| N9 | Two writers at the same time are not detected (only writes during searches are) | Edge case | **Done** in phase 3: changes are serialized by the index lock |
| N10 | 32-bit platforms are untested and probably unsupported: document or reject at build time | Edge case | Next hardening pass |

## Part 3: Order of work

| Phase | Contents | Branch | Effort |
|---|---|---|---|
| **0. Measure** | SIFT1M (1 million × 128) and GloVe (1.2 million × 100, angular); queries per second, p50 and p99 latency, build time, memory; hnswlib on the same machine with the same settings; a real profiler (perf on Linux, VTune or Windows Performance Analyzer on Windows). The baseline is recorded in [Performance and tuning](performance.md) | `perf/benchmarks` | 1–2 days |
| **1. Harden** | Test plan for E1–E13, then the tests and fixes | `fix/edge-cases` | 1–2 days |
| **2. Quick wins** | O1, O2, O3, O4 and O8, each kept only if measured faster | `perf/search-quick-wins` | 2–3 days |
| **3. Concurrency stage A** | One lock around the index: removes the biggest open risk | `feature/concurrent-writes-a` | Half a day |
| **4. Layout decision** | O5 and O6, if Phase 0 and 2 numbers justify them; skipped otherwise | `perf/node-layout`, `feature/sq8` | 4–8 days if both |
| **5. Persistence** | Save and load (scenarios P1–P42 of the core plan), with one file format for the final layout and metadata | `feature/persistence` | 3–4 days |
| **6. Parallel build** | O7 (concurrency stage B) and O9 | `feature/concurrent-writes-b` | 4–6 days |
| **7. Filtered and Flat speed** | O10, O11 and O12, each only if measurements show the need | One branch each | As needed |

Each phase ends with a merged pull request, passing CI, and updated documentation, before the next begins.

## Rules for every phase

- **Test plan first,** with scenario IDs, before implementation, as for every feature so far.
- **Benchmark before and after.** Only changes at least **5% faster** are kept (decided for Phase 2). Timings on a shared machine vary by up to 30% between identical runs, so gains are judged on a quiet machine with dedicated cores, running the baseline and the candidate alternately several times and comparing medians. Each phase records its numbers in [Performance and tuning](performance.md).
- **Recall never gets worse:** recall at each `ef` stays within noise of the baseline. Changes that do not alter the search order give identical results, which the determinism tests check.
- **No guarantee is lost:** all-or-nothing behavior under out-of-memory (sweeps for every new allocation path), ThreadSanitizer cleanliness, the coverage gate, and zero warnings under strict compiler settings.
- **Fresh builds for verification.** Final checks use builds from scratch, never incremental ones, after a stale build misled an earlier verification.

## Decisions needed

D1 and D2 were applied in Phase 1; D3 to D5 have been decided and apply when their phases start.

| ID | Question | Options | Recommendation |
|---|---|---|---|
| D1 | Must the same seed give the same graph on GCC, Clang and MSVC (E1)? | Portable across platforms, or same platform only | **Applied: portable levels**, checked in CI |
| D2 | May a predicate call back into the index (E2, O3)? | Support re-entrant searches; or forbid it, documented and detected where possible | **Applied: searches allowed, changes refused** with `std::logic_error`. Phase 2's scratch space (O3) must keep searches re-entrant |
| D3 | Must Flat batch results stay identical to separate searches (O12)? | Identical, using the matrix kernel only to find candidates that are then re-checked exactly; or allow rounding differences | **Decided: the user chooses.** An exact mode (the default) gives results identical to separate searches; a fast mode uses the matrix kernel and accepts tiny rounding differences |
| D4 | Pursue SQ8 (O6)? | An optional per-index setting; or not until Phase 0 shows memory or bandwidth is the bottleneck | **Decided: offer it as an optional per-index setting** in Phase 4; full-precision floats stay the default |
| D5 | Must parallel builds be reproducible (O7)? | A reproducible single-threaded mode is enough; or parallel builds reproducible too | **Decided: single-threaded mode is enough.** Parallel builds may differ between runs; a single-threaded mode always gives the identical graph |

## Phase 1 results

Decisions applied: **D1** (levels identical on every platform; a fixed-answer test now checks it on every CI platform) and **D2** (predicates may search the index, including batch searches, but not change it). The test group `hardening` covers each edge case:

| Edge case | Test | Finding | Outcome |
|---|---|---|---|
| E1 | `h01` | The level draw was already portable. The real cross-machine difference is the SIMD kernels, which round distances differently | Documented precisely; fixed-answer levels checked on every platform |
| E2 | `h02`, `h03`, `h14` | **Confirmed: a nested batch hung** (deadlock on the shared pool); changes from a predicate were undetected | Nested `parallel_for` runs inline; every index refuses changes during a search with `std::logic_error` |
| E3 | `h04` | **Confirmed:** 8 of 2,000 cosine self-distances were negative; radius-0 range search found exact copies only 89% of the time | Cosine distances clamped at 0 inside the indexes (the public kernels are unchanged); a small radius is documented for exact copies |
| E4 | `h05` | `M = 2,000,000` attempted a huge allocation (`bad_alloc`) | `M` limited to 2048 with a clear error |
| E5 | `h06` | IDs 0 and 2^64−1 already worked | Test added |
| E6 | `h07` | Vectors taken from the index's own storage already worked | Test added |
| E7 | `h08` | Dimension 0 was already rejected | Test added |
| E8 | `h09` | 2,000 dynamic fields worked correctly | Test added |
| E9 | `h10` | **Slow:** 136 ms for a 5,000-tag filter over 20,000 vectors | Tags sorted and binary-searched: 4.7 ms (29 times faster) |
| E10 | `h11` | A throwing predicate in a threaded batch already behaved correctly | Test added |
| E11 | (none) | Batch memory is 4 bytes per slot per thread | Documented in the user guide |
| E12 | `h12` | Correct results; no slowdown on the CPU tested (0.50 ms against 0.66 ms) | Test added; older CPUs noted in the user guide |
| E13 | `h13` | The deepest filter works on pool threads | Test added (runs on Windows in CI) |

## Status

| Phase | Status |
|---|---|
| 0. Measure | **Baseline done here** on generated data (see [Performance and tuning](performance.md#baseline-for-performance-work-phase-0)); real datasets still to run on a quiet machine |
| 1. Harden | **Done** (see Phase 1 results) |
| 2. Quick wins | Not started |
| 3. Concurrency stage A (one lock) | **Done** (see [Thread safety](architecture.md#thread-safety)) |
| 4. Layout decision | SQ8 part **done** (decision D4; see the [compression and index types plan](compression-and-index-types-plan.md)); node layout not started |
| 5. Persistence | **Done** (file format version 1, now version 3; see [Architecture and design](architecture.md#saving-and-loading)) |
| 6. Parallel build | **Done** (`add_batch`; see [Parallel building](architecture.md#parallel-building-add_batch)) |
| 7. Filtered and Flat speed | Not started |