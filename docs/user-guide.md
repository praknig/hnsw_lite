# User guide

How to use everything hnsw-lite offers. For building and setup, see [Getting started](getting-started.md); for every class and function, see the [API reference](api-reference.md).

All names below are in the `vecdb` namespace. Code examples assume `using namespace vecdb;`.

## Contents

- [Core ideas](#core-ideas)
- [Creating an index](#creating-an-index)
- [Adding vectors](#adding-vectors)
  - [Adding many vectors at once](#adding-many-vectors-at-once)
- [Searching](#searching)
- [Updating vectors](#updating-vectors)
- [Removing vectors and compacting](#removing-vectors-and-compacting)
- [Metadata](#metadata)
- [Filtered search](#filtered-search)
- [Batch search](#batch-search)
- [Range search](#range-search)
- [Reading records back](#reading-records-back)
- [Searching by stored ID](#searching-by-stored-id)
- [Deadlines, cancellation and progress](#deadlines-cancellation-and-progress)
- [Changing many vectors by filter](#changing-many-vectors-by-filter)
- [Compressing vectors (SQ8)](#compressing-vectors-sq8)
- [Binary quantization](#binary-quantization)
- [Disk-based index](#disk-based-index)
- [Inverted file (IVF)](#inverted-file-ivf)
- [Product quantization (PQ)](#product-quantization-pq)
- [Shorter vectors (Matryoshka truncation)](#shorter-vectors-matryoshka-truncation)
- [Keeping only the stored vectors](#keeping-only-the-stored-vectors)
- [Keeping only codes (SQ8 without full precision)](#keeping-only-codes-sq8-without-full-precision)
- [Saving and loading](#saving-and-loading)
- [Threads](#threads)
- [Errors and guarantees](#errors-and-guarantees)
- [Memory](#memory)

## Core ideas

- **A vector** is a fixed-length list of floats, such as an embedding produced by a machine-learning model. Every vector in an index has the same length, the **dimension**.
- **Your IDs:** each vector is stored under a 64-bit integer you choose, and searches return those IDs. Use them to find the matching document, image or row in your own data.
- **Nearest neighbors:** a search returns the `k` stored vectors closest to a query vector, closest first.
- **Two indexes:**
  - `FlatIndex` compares the query with every stored vector. Results are **exact**, but the time grows in proportion to the number of vectors.
  - `HnswIndex` builds a multi-level graph and walks it towards the query. It is **approximate** (it can occasionally miss a true neighbor) but far faster on large collections. How often it finds the true neighbors is called **recall**.

## Creating an index

```cpp
FlatIndex exact(768, Metric::Cosine);   // 768-dimensional vectors
HnswIndex fast(768, Metric::Cosine);    // default settings
```

### Choosing a metric

| Metric | Use when | The `distance` value returned |
|---|---|---|
| `Metric::Cosine` | Text embeddings and most model outputs: direction matters, length does not | 1 minus the cosine similarity: 0 for the same direction, up to 2 for opposite |
| `Metric::L2` | Coordinates, or any data where actual distance matters | The **squared** Euclidean distance (no square root, which is faster and ranks identically) |
| `Metric::InnerProduct` | Embeddings trained for dot-product scoring | The **negative** dot product, so a larger dot product means a smaller distance |

Every metric follows the same rule: **smaller distance means closer**, and cosine distances are never negative. With cosine, vectors are normalized automatically when added and when searched, so you do not need to normalize them yourself. An all-zero vector has no direction: with cosine it is accepted and sits at distance 1 from everything, as in hnswlib and Faiss.

Inner product is not a true distance, and HNSW recall is lower with it on vectors of very different lengths. If you can, normalize your vectors and use cosine.

### HNSW settings

```cpp
HnswParams params;
params.M = 16;                  // links per node (2 to 2048); more = better recall, more memory
params.ef_construction = 200;   // effort spent building; more = better graph, slower inserts
params.seed = 42;               // same seed and operations = identical graph (see below)
params.repair_on_remove = true; // repair links around removed vectors
HnswIndex index(768, Metric::Cosine, params);
```

The defaults suit most data. `M` between 8 and 48 is typical; values below 8 leave parts of the graph poorly connected, and values above 2,048 are rejected.

**Reproducibility:** the same seed and the same operations give an identical graph on the same machine. Random levels are identical on every platform, but the SIMD distance code differs by CPU (AVX-512, AVX2, NEON or plain C++) and rounds distances slightly differently, so graphs built on different machines can differ a little. See [Performance and tuning](performance.md) for measured trade-offs.

## Adding vectors

```cpp
index.add(42, embedding);   // embedding: std::vector<float>, std::array, or any contiguous range of floats
```

- The vector must have exactly the index's dimension.
- NaN and infinite values are rejected.
- An ID can only be stored once at a time. Adding an ID that is already stored throws; use `remove` first to replace a vector.
- `add` copies the vector, so your own buffer can be reused or freed afterwards.
- Values extremely close to zero (subnormal floats, below about 1e-38) give correct results. They caused no slowdown on the CPUs tested, but some older CPUs process them many times slower; if you see that, round such values to 0.

A failed `add` leaves every stored vector, its metadata and all search results as they were. See [Errors and guarantees](#errors-and-guarantees) for the details.

### Adding many vectors at once

```cpp
std::vector<std::uint64_t> ids = {1, 2, 3};
std::vector<float> vectors = ...;           // ids.size() * dim() values, back to back
index.add_batch(ids, vectors);              // all cores (HnswIndex)
index.add_batch(ids, vectors, metadata, 8); // with one Metadata per vector, on 8 threads
```

- **All or nothing:** every vector, ID (against the index and within the batch) and metadata entry is checked first; any problem throws `std::invalid_argument` and changes nothing. With a dynamic schema, a field that is new must have the same type throughout the batch.
- **HNSW links vectors on several threads** (`threads = 0`, the default, uses all cores). Vectors that reuse slots of removed vectors are linked one at a time first, exactly as `add` would; the rest in parallel.
- **One thread gives exactly the same index as calling `add` for each vector in order.** With more threads the graph differs slightly from run to run (which vector links first depends on timing), while search quality is the same.
- **Searches wait** while a batch is added, then see all of it or none of it.
- **If memory runs out**, an HNSW index removes the batch's vectors again (repairing the graph around them when memory allows): none of the batch remains, and every earlier vector is still stored and searchable. A Flat index is restored exactly. Either way, the exception is passed on.
- On `FlatIndex`, `add_batch` has the same checks and guarantees; inserting is a copy, so it runs on the calling thread.

## Searching

```cpp
std::vector<SearchResult> results = flat.search(query, 10);       // FlatIndex: k
std::vector<SearchResult> results = hnsw.search(query, 10, 64);   // HnswIndex: k, ef
for (const SearchResult& r : results) use(r.id, r.distance);
```

- Results are sorted closest first. Asking for more results than there are vectors returns all of them.
- **`ef` (HNSW only)** controls how many candidates the search keeps while walking the graph. It is the speed-versus-accuracy dial: 64 (the default) is a good start; raise it if recall matters more than speed. It is always at least `k`.
- With equal distances, results are ordered by an internal position, so the order of ties is stable but not by ID.

Useful queries about the index: `size()` (vectors stored), `contains(id)`, `dim()`, `metric()`.

## Updating vectors

```cpp
bool found = index.update(42, new_embedding);   // false if 42 is not stored
bool added = index.upsert(42, new_embedding);   // adds 42 if new, replaces it otherwise; true if added
```

- `update` and `upsert` keep the vector's metadata. To replace the vector and its metadata together, in one step, pass the metadata too: `index.update(42, new_embedding, Metadata().set("year", 2025))`. Fields not given are then cleared; to change only some fields, use `set_metadata`.
- The new vector is checked first: a wrong dimension, NaN or infinity throws `std::invalid_argument`, even for an unknown ID.
- **An update never loses the old vector.** `HnswIndex` builds and links the new version in another slot while the old one stays searchable, and only then switches the ID over. If anything fails, including running out of memory, the old vector is exactly as it was. The switch uses at most one extra slot, which later updates and inserts reuse.
- `FlatIndex` simply overwrites the vector in place.

## Removing vectors and compacting

```cpp
bool removed = index.remove(42);   // false if 42 was not stored
index.add(42, new_embedding);      // the ID can be used again immediately
```

- `FlatIndex` removes for real: the last vector moves into the removed one's place. Nothing to clean up later.
- `HnswIndex` frees the vector's ID and reuses its memory for the next insert. The graph is repaired around the removed vector so searches stay accurate. `deleted_count()` reports removed slots not yet reused, and `capacity()` the slots in use plus those.
- **`compact()`** rebuilds an HNSW index from only its live vectors, giving the best graph quality and returning memory. It is worth calling after removing a large share of the vectors (say a third), at a quiet moment: it takes about as long as building the index from scratch. If it runs out of memory, the original index is kept unchanged.

## Metadata

Each vector can carry named fields, used for filtering and readable with `get_metadata`.

| Type | C++ value | Example |
|---|---|---|
| `FieldType::Int` | Any integer (stored as 64-bit) | `.set("year", 2024)` |
| `FieldType::Float` | `double` (NaN and infinity rejected) | `.set("price", 9.99)` |
| `FieldType::Bool` | `bool` | `.set("in_stock", true)` |
| `FieldType::Keyword` | String | `.set("topic", "tech")` |
| `FieldType::Tags` | A set of strings | `.set_tags("tags", {"ai", "chips"})` |

```cpp
index.add(7, embedding, Metadata().set("topic", "tech").set("year", 2024).set_tags("tags", {"ai"}));

std::optional<Metadata> md = index.get_metadata(7);    // nothing if 7 is not stored
std::int64_t year = md->get("year")->i;                // .i, .f, .b, .s or .tags, by type

index.set_metadata(7, Metadata().set("year", 2025));   // changes only the given fields
index.set_metadata(7, Metadata().unset("tags"));       // removes a field
```

### Schemas

The schema decides which fields an index accepts. Pass it to the constructor:

```cpp
// Strict: only these fields. Unknown fields and wrong types are rejected at once,
// and filters on unknown fields are errors, so typos are caught early.
Schema strict = Schema::strict({{"topic", FieldType::Keyword}, {"year", FieldType::Int}});
HnswIndex index(768, Metric::Cosine, HnswParams{}, strict);
FlatIndex flat(768, Metric::Cosine, strict);

// Dynamic (the default): fields are created the first time they are used, with
// the type of their first value. You can still declare some fields up front.
Schema dynamic = Schema::dynamic({{"year", FieldType::Int}});
```

In both modes a field keeps one type forever: storing text in an integer field is rejected. An integer is accepted for a float field.

Strings are stored once, however many vectors use them, and strings no vector uses any more are reclaimed automatically. Fields themselves are never removed, even if no vector uses them, because that would silently forget their type.

## Filtered search

### Writing filters

```cpp
Filter f = Filter::eq("topic", "tech") && (Filter::ge("year", 2020) || Filter::has_tag("tags", "ai"));
```

| Condition | Works on | Meaning |
|---|---|---|
| `eq`, `ne` | All types except tags | Equal, not equal |
| `lt`, `le`, `gt`, `ge` | Integers and floats | Less than, at most, greater than, at least |
| `between(field, lo, hi)` | Integers and floats | `lo <= value <= hi` |
| `in(field, {a, b, c})` | Integers, floats, keywords, booleans | Equal to any of the values |
| `has_tag`, `has_any_tag`, `has_all_tags` | Tags | Contains the tag, any of them, all of them |
| `exists(field)` | All types | The vector has a value for the field |
| `&&`, `\|\|`, `!` | Filters | And, or, not |

Rules worth knowing:

- **Missing fields:** a comparison never matches a vector that lacks the field, including `ne`. Use `!` to include them: `!Filter::eq("topic", "tech")` matches everything not about tech, including vectors with no topic.
- **Numbers:** integers and floats compare exactly with each other.
- **Nesting** is limited to 256 levels. To combine many alternatives, use `in` or `has_any_tag`, which are also faster.
- Filters are checked against the index's schema when used: a type mismatch (comparing a keyword with a number, say) throws `std::invalid_argument`.

### Searching with a filter

```cpp
SearchOptions options;
options.filter = Filter::eq("topic", "tech") && Filter::ge("year", 2020);
auto results = hnsw.search(query, 10, 64, options);
auto exact   = flat.search(query, 10, options);
```

Results never include a vector that fails the filter. If fewer than `k` vectors match, you get only those.

### Conditions outside the index: predicates

When the condition lives elsewhere (permissions in your database, for example), pass any function of the ID:

```cpp
SearchOptions options;
options.predicate = [&](std::uint64_t id) { return user_can_see(id); };
```

A predicate can be combined with a filter: both must pass, and the filter is checked first. In batch search the predicate is called from several threads at once, so it must be safe to call concurrently.

A predicate may **search** the index it filters, including a batch search, but must not **change** it: a change made during a search throws `std::logic_error` and leaves the index unchanged.

### How filtered HNSW searches work

A filtered HNSW search either scans the matching vectors exactly, or walks the graph while skipping vectors that fail the filter. A **query planner** chooses per query by estimating how many vectors match:

- **Few matches:** exact scan. This is the default when at most 2,000 vectors, or at most 1%, match.
- **Many matches:** graph search, automatically searching wider (a larger `ef`) because some of the vectors it visits do not count.

You can see and steer the decision:

```cpp
SearchStats stats;
options.stats = &stats;                       // filled in by the search
options.strategy = Strategy::ForceExact;      // or ForceGraph; Auto (default) lets the planner choose
options.planner = PlannerParams{5000, 0.05};  // thresholds for this query: max matches, max fraction

index.set_planner_params(PlannerParams{5000, 0.05});  // or for every query on this index
index.create_payload_index("topic");          // exact match counts for a frequently filtered field
```

`SearchStats` reports the strategy used and why, the estimated matches, the `ef` used, and how many vectors were visited. A **payload index** keeps exact per-value counts for a keyword, boolean or tag field, so the planner counts matches instead of estimating them.

## Batch search

Answer many queries in one call. Put the queries back to back in one buffer:

```cpp
std::vector<float> queries = /* query 0 (dim floats), then query 1, ... */;
auto all  = hnsw.search_batch(queries, /*k=*/10, /*ef=*/64, /*threads=*/0);   // 0 = one per CPU core
auto all2 = flat.search_batch(queries, /*k=*/10, /*threads=*/4);
// all[i] holds the results of query i
```

- Results are identical to separate `search` calls, whatever the thread count.
- Every query is checked before any work starts; one invalid query rejects the whole batch.
- An optional `SearchOptions` (filter, predicate) applies to all queries. Statistics are not filled in for batches.
- Thread counts are capped at a sensible maximum (64, or 4 per core on larger machines).
- Each thread holds a visited list of 4 bytes per vector slot, so a batch on 64 threads over 1 million vectors uses about 256 MB extra while it runs. Use fewer threads if memory is tight.

## Range search

Return every vector within a distance of the query, instead of a fixed number:

```cpp
auto nearby = flat.search_range(query, l2_radius(0.5f));               // L2: everything within distance 0.5
auto close  = hnsw.search_range(query, 0.2f);                          // cosine: within distance 0.2
auto capped = hnsw.search_range(query, 0.2f, /*max_results=*/100);     // at most the 100 closest
```

- The radius uses the same distance values searches return. For L2 that is the **squared** distance, so convert an ordinary radius with `l2_radius(r)`. For inner product, distances are negative.
- With cosine, rounding can give an exact copy of the query a distance of about 0.0000001 instead of 0, so use a small radius such as `1e-6f`, rather than 0, to find exact copies.
- Without `max_results`, there is no limit: a large radius can return the whole index.
- `FlatIndex` is exact. `HnswIndex` is approximate, like its normal search; pass `Strategy::ForceExact` in the options when every vector within the radius must be found.
- Filters and predicates work here too.

## Reading records back

```cpp
std::optional<std::vector<float>> v = index.get_vector(42);   // exactly as given to add() or update()
std::optional<Record> r = index.get(42);                      // r->id, r->vector, r->metadata

ScanOptions o;
o.filter = Filter::eq("category", "tech");   // optional
o.limit = 100;                                // records per page
for (Page page = index.scan(o); ; page = index.scan(o)) {
    for (const Record& rec : page.records) { /* ... */ }
    if (!page.next) break;
    o.after = page.next;                      // continue after the last ID returned
}
```

- **`get_vector` returns the vector exactly as it was given**, bit for bit. For cosine indexes, which normalize vectors for searching, the original is kept for this purpose, which **doubles the memory used for vectors** in cosine indexes.
- **`scan` lists records in ascending ID order**, a page at a time, optionally only those matching a filter (same rules as `count_where`). Each page continues after the last ID of the previous one, so changes between pages never repeat a record or skip one that stays stored; records added later with a higher ID appear, removed ones don't.
- **Set `with_vectors` or `with_metadata` to false** to leave those fields empty, for example to list only IDs.
- **Each page costs one pass over the index**, which suits exports and administration; for finding similar vectors, use `search`.
- All three are reads: they run alongside searches and are allowed inside them.

**Ties.** Results with exactly equal distances (for example, copies of the same vector) come in an unspecified order, which may differ between HNSW and Flat and change after `compact()`. When a tie straddles the k-th place, any of the tied vectors is a correct answer.

## Searching by stored ID

```cpp
auto similar = index.search_by_id(42, 10);              // "more like record 42", without record 42
auto with_it = index.search_by_id(42, 10, 64, {}, true); // including record 42 itself
if (similar) for (const SearchResult& r : *similar) { /* ... */ }
```

- It searches with the vector stored for the ID, under one lock, so the record cannot change in between. Filters, predicates, deadlines and every other search option work as in `search`.
- It returns nothing (an empty `std::optional`) if the ID is not stored.
- The record itself is left out unless the last argument is `true`; other records with an identical vector are kept.

**A C++ trap with functions returning `std::optional`** (`search_by_id`, `get_vector`, `get`): do not loop directly over `*index.search_by_id(...)`. A range-based `for` keeps only its outermost value alive, here a reference into the temporary optional, which is destroyed before the loop body runs (fixed in C++23). Store the result in a variable first, as above.

## Deadlines, cancellation and progress

```cpp
SearchOptions o;
o.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
SearchStats stats;
o.stats = &stats;
auto results = index.search(query, 10, 64, o);   // best found within 20 ms
if (stats.stopped_early) { /* results may be incomplete */ }

std::atomic<bool> cancel{false};                 // set to true from any thread
OperationOptions op;
op.cancel = &cancel;
op.progress = [](std::size_t done, std::size_t total) { std::printf("%zu/%zu\n", done, total); };
try {
    index.add_batch(ids, vectors, 0, op);
} catch (const vecdb::Cancelled&) { /* undone: the index is as before */ }
```

- **Searches** (single, batch, range and by ID) take a `deadline` and a `cancel` flag in `SearchOptions`. When either fires, the search stops and returns the best results found so far, sorted and valid but possibly incomplete, and sets `SearchStats::stopped_early`. They are checked every few dozen steps, so a search stops within microseconds. A search started inside a predicate has its own deadline, not its caller's.
- **`add_batch`, `compact` and `save`** take `OperationOptions`: a `deadline`, a `cancel` flag and a `progress` callback. A cancelled or late operation is **undone** and throws `vecdb::Cancelled`: a batch is rolled back, a compaction discarded, a save leaves any existing file untouched and no temporary file (a stream may hold an incomplete index).
- **Progress** is reported as `(done, total)`: from 0, never decreasing, at most about 100 times, ending at `(total, total)` on success. Calls never overlap but may come from worker threads, and the callback must not use the index. If it throws, the operation is undone and its exception passed on.

## Changing many vectors by filter

```cpp
using vecdb::Filter;
index.remove_batch(std::vector<std::uint64_t>{4, 8, 15});              // by ID: unknown IDs are skipped
std::size_t n = index.count_where(Filter::eq("category", "old"));     // preview: how many match
index.remove_where(Filter::eq("category", "old"));                    // like SQL's DELETE ... WHERE
index.set_metadata_where(Filter::lt("year", 2000),                    // like SQL's UPDATE ... WHERE
                         Metadata().set("archived", true).unset("featured"));
```

The three `_where` functions take the same filters as searches; all four return how many vectors matched. `remove_batch` skips unknown and repeated IDs, and like `remove_where` always completes.

- **The filter is evaluated once, before anything changes.** So `set_metadata_where` applies to each matching vector exactly once, even when it changes the very field the filter tests (for example renaming a category).
- **A filter naming an unknown field** throws `std::invalid_argument` under a strict schema, and matches nothing under a dynamic one, as in searches.
- **`remove_where` always completes:** once the filter is accepted, every match is removed. If memory runs out while repairing the HNSW graph around them, the repair is cut short (as by `remove`); no match is left behind.
- **`set_metadata_where` is all or nothing:** invalid changes throw `std::invalid_argument`, even when nothing matches, and if memory runs out no vector has changed. It works on a copy of the metadata and swaps it in when complete, so it briefly needs extra memory about the size of the index's metadata.
- **Searches see the index before or after a change, never in between**, and like all changes these cannot be made from inside a search (`std::logic_error`); `count_where` is a read and can.

## Compressing vectors (SQ8)

```cpp
HnswParams p;
p.quantization.codec = CodecKind::SQ8;   // one byte per dimension
p.quantization.train_after = 1000;       // train automatically at 1,000 vectors
HnswIndex index(768, Metric::Cosine, p);

FlatIndex flat(768, Metric::L2, Schema::dynamic(), p.quantization);
flat.train_codec();                      // or train now, on the stored vectors
```

- **SQ8** stores each dimension as one of 256 levels between its smallest and largest trained value. Searches compare the query with these codes, then **re-rank** the best candidates at full precision (`rerank`, on by default). In tests, recall with re-ranking matched full precision exactly; with codes alone (`rerank = false`) it was about 1% lower.
- **Training** happens automatically once `train_after` vectors are stored (0: only by `train_codec()`), and again whenever you call `train_codec()`. Until then, searches use full precision exactly as without a codec. Training is all or nothing; if memory runs out during automatic training, the add still completes and training is retried on a later add.
- **Memory:** full-precision vectors are kept too (for re-ranking and `get_vector`), so for now SQ8 adds 25% to vector memory rather than saving it. The saving comes with full-precision vectors kept on disk; see the [plan](compression-and-index-types-plan.md).
- Range searches use full precision. Batch, filtered and exact searches use codes. Saved files keep the codec, its training and the codes (file format version 3).
- **Setting `HnswParams` positionally** (`HnswParams{16, 200, 42}`) leaves out `quantization`, and compilers warn about that with `-Wextra`. Set fields by name instead, as above.

## Binary quantization

```cpp
HnswParams p;
p.quantization.codec = CodecKind::Binary;  // one bit per dimension
p.quantization.rerank_factor = 32;         // binary needs many candidates re-ranked
HnswIndex index(768, Metric::Cosine, p);
auto results = index.search(query, 10, 400);  // HNSW re-ranks its whole beam: use a larger ef
```

- **One bit per dimension**, 32 times smaller than floats. Each dimension gets a threshold (its mean in the training sample); a bit records whether a value lies above it. Distances compare the full-precision query with each code's reconstruction (threshold plus or minus the dimension's average spread).
- **Binary codes are coarse, so re-ranking at full precision is essential, with many candidates.** Measured recall@10 on 3,000 clustered vectors: Flat with `rerank_factor` 4 (the default, suited to SQ8) reached 0.63 to 0.73, with 16 about 0.86, and with 32 0.96 to 0.99; HNSW re-ranks its whole beam, and reached 0.74 to 0.79 at `ef` 64 and 0.98 to 0.99 at `ef` 400.
- **Keep the full vectors** (the default). Without them (`keep_full_vectors = false`) nothing can be re-ranked: recall was 0.34 to 0.39, and at low dimensions many vectors share the same code (12 dimensions allow only 4,096 codes), which also weakens the graph.
- Training, saving, truncation and everything else work as for SQ8.
- Code distances use SIMD instructions (AVX2, AVX-512, NEON) when the CPU has them, like full-precision distances; see [Performance and tuning](performance.md#compression).

## Disk-based index

```cpp
// Build once (the vectors must fit in memory while building), then search from disk.
DiskIndexParams params;
params.graph.R = 64;      // neighbors per node
params.graph.L = 100;     // beam while building
DiskIndex::build("vectors.idx", ids, vectors, 768, Metric::Cosine, params, metadata);

auto index = DiskIndex::open("vectors.idx");  // loads only PQ codes, IDs and metadata
SearchOptions options;
options.filter = Filter::eq("lang", "en");
DiskSearchStats stats;
auto results = index->search(query, 10, DiskSearchParams{100, 4}, options, &stats);
// stats.nodes_read: disk reads for this query
```

- **For indexes larger than memory** (DiskANN-style): the graph and the full vectors stay on disk in 4 KB blocks, one read per node; memory holds only compact PQ codes (by default one byte per 2 dimensions, an eighth of the floats), the IDs and the metadata.
- **Searching:** PQ codes rank the candidates; the best unread ones are read from disk, `beam_width` (W) per round, their exact distances computed from the vectors read, their neighbors added; the search ends when the best `L` candidates were all read. Results carry exact distances. Measured on 5,000 vectors of 32 dimensions: `L` 50 read 56 nodes (1.1%) for recall@10 0.98, `L` 100 read 105 for 1.00. Raise `L` for recall, `W` for fewer round trips (W 4 needed 3 times fewer rounds than W 1, for the same reads).
- **The PQ codes guide which nodes are read**, so they are finer than PQ's general default: with one part per 8 dimensions, recall at `L` 50 fell to 0.71 for the same number of reads. Set `DiskIndexParams::pq_subspaces` to trade memory for fewer wasted reads.
- **Filters and predicates:** only matching vectors are returned, never fewer than `k` if `k` match. A filter matching at most `L` vectors reads exactly those (an exact answer); a broader one searches the graph, widening `L` until it finds `k`.
- **Records:** `contains`, `get_vector`, `get`, `scan`, `search_by_id` and `search_batch` work as for the in-memory indexes; vectors come back as stored (normalized for cosine). Deadlines and cancellation stop a search with the best results so far. Safe to search from many threads.
- **Building larger than memory:** `write_vectors_file("vectors.fbin", vectors, dim)` (or any `.fbin` file: a 32-bit count, a 32-bit dimension, then the floats, as DiskANN and the big-ANN benchmarks use), then `LargeBuildParams build; build.memory_budget = 8ull << 30; DiskIndex::build_from_file("vectors.idx", "vectors.fbin", Metric::Cosine, build);`. If the vectors fit the budget it builds in memory (the same file as `build`); otherwise it divides them into partitions that fit (each vector joins its 2 nearest), builds a graph per partition, and merges them in one pass. No partition ever exceeds the budget; PQ codes, IDs and metadata (what a search holds in memory) come on top. Measured on 6,000 vectors built from 24 partitions: every vector reachable, recall@10 0.998 at `L` 100 (the same as an in-memory build) and 0.950 at `L` 50 (0.983 in memory): give partitioned builds a slightly larger `L`.
- **Concurrent reads:** `DiskOpenOptions io; io.io_threads = 8; DiskIndex::open(path, io)` reads the W nodes of each round at the same time, so a round costs about one read's latency instead of W. Results are identical either way; several searches share the threads. It helps on storage with real latency (network disks, cloud volumes, hard disks); on a fast local SSD or with the file cached by the operating system, gains are small. Measured with a simulated 3 ms per read, W 4: 194 ms instead of 679 for the same 60 rounds.
- **Read-only:** to change it, rebuild. L2 and cosine only. Building writes a temporary file and replaces `path` only once complete. Every block is checksummed and checked when read: damage is reported as `FormatError` where it is met.

## Inverted file (IVF)

```cpp
IvfParams ivf;
ivf.lists = 1024;         // about the square root of the number of vectors is a common start
ivf.probes = 32;          // lists searched per query: the speed and recall trade-off
ivf.train_after = 50000;  // train automatically (or call train_lists())
FlatIndex index(768, Metric::Cosine, Schema::dynamic(), QuantizationParams{}, ivf);

SearchOptions o;
o.probes = 64;            // more lists for this search only
auto results = index.search(query, 10, o);
```

- **k-means divides the vectors into lists**, and a search scans only the lists nearest to the query instead of everything. It is an option of `FlatIndex`, so filters, metadata, deadlines, codecs, saving and everything else work as before.
- **Probes trade speed for recall.** Measured on noisy synthetic data with 64 lists: 1 probe scanned 1.5% of the vectors for recall@10 of 0.16, 16 probes 25% for 0.81, all 64 probes everything for exact results. Real embeddings, which cluster better, usually need fewer probes.
- **Never fewer than k results:** if the probed lists hold fewer than `k` matching vectors (with a selective filter, say), more lists are probed.
- **IVF-PQ:** combine with `QuantizationParams{.codec = CodecKind::PQ}` for small codes and fast scans; re-ranking works as for PQ.
- **Until trained**, searches scan everything. Training (`train_lists()`, or automatically at `train_after` vectors and at least `lists`) is all or nothing, and can be repeated. **Training costs** about lists × sampled vectors × dimensions × iterations (up to 25; at most 100,000 vectors are sampled): 1,024 lists over 50,000 vectors of 768 dimensions is around 10^12 operations, minutes on a few cores. k-means uses every core; train once on a representative sample. Vectors added later go to their nearest list. Range searches always scan everything.

## Product quantization (PQ)

```cpp
HnswParams p;
p.quantization.codec = CodecKind::PQ;
p.quantization.pq_subspaces = 96;           // 768 dimensions in 96 parts of 8: 96 bytes per vector
p.quantization.train_after = 10000;         // k-means needs a representative sample
HnswIndex index(768, Metric::Cosine, p);
auto results = index.search(query, 10, 200);  // re-ranked: give the search a generous beam
```

- **Each vector is split into parts**, and each part is stored as the index of its nearest of 256 centroids, learned by k-means: one byte per part. With the default (`pq_subspaces = 0`: one part per 8 dimensions) that is 32 times smaller than floats. Dimensions that do not divide evenly go to the first parts.
- **Distances are table lookups:** a query computes its distance to every centroid once, then each code costs one lookup per part.
- **Re-ranking is essential, as with binary codes.** Measured recall@10 on synthetic clustered data (64 dimensions, 8 per part): HNSW 0.70 to 0.79 at `ef` 64 and 0.92 to 0.97 at `ef` 200; Flat 0.62 to 0.74 with `rerank_factor` 4 and 0.89 to 0.95 with 16; codes only 0.31 to 0.40. More parts (fewer dimensions each) give better recall for more bytes; real embeddings usually suit PQ better than this noisy test data.
- **Training** runs k-means on each part (up to 100,000 sample vectors), so it takes longer than for SQ8 or binary codes; with fewer than 256 training vectors there are fewer centroids. Training, saving, truncation and codes only work as for SQ8.

## Shorter vectors (Matryoshka truncation)

```cpp
HnswParams p;
p.quantization.truncate_to = 256;        // index only the first 256 of 768 dimensions
HnswIndex index(768, Metric::Cosine, p);
index.add(id, full_vector);              // vectors and queries are still given in full (768 values)
auto results = index.search(full_query, 10);
auto v = index.get_vector(id);           // all 768 values, exactly as given
```

- **Only for embeddings trained for it**, such as OpenAI's `text-embedding-3` models or Nomic Embed: these put the most important information first, so a prefix of the vector is still a good, slightly less precise embedding. With other embeddings, truncation damages results.
- The index stores and searches only the first `truncate_to` values (re-normalized for cosine), so each distance costs proportionally less and building is faster. `dim()` stays the full dimension; `indexed_dim()` reports the truncated one.
- Inputs are checked in full: a vector of the wrong size, or with NaN or infinity anywhere (even in the part that is cut off), is rejected.
- **Memory:** by default the full vectors are kept too, so that `get_vector` returns them, which uses more memory, not less. Set `keep_full_vectors = false` (below) to keep only the truncated vectors: about a third of the memory for 768 to 256.
- It combines with SQ8 (the codes then cover the first `truncate_to` dimensions), and is saved with the index (file format version 3). `0`, or the full dimension, means no truncation.

## Keeping only the stored vectors

```cpp
HnswParams p;
p.quantization.truncate_to = 256;
p.quantization.keep_full_vectors = false;  // keep only the 256 stored values
HnswIndex index(768, Metric::Cosine, p);
index.get_vector(id);                      // 256 values (normalized, for cosine)
```

Cosine indexes (which normalize vectors) and truncating indexes keep a second copy of each vector exactly as given, so that `get_vector` can return it. `keep_full_vectors = false` drops that copy:

- **Memory:** a cosine index stores about half as much; a truncating index only the truncated vectors.
- **`get_vector`, `get` and `scan` return the stored vector:** normalized for cosine, and the first `truncate_to` values when truncating. Search results are identical, since the stored vectors are the same.
- It changes nothing for L2 and inner-product indexes without truncation or a codec, which keep no second copy. With SQ8 it goes further: see the next section.
- It is saved with the index.

## Keeping only codes (SQ8 without full precision)

```cpp
HnswParams p;
p.quantization.codec = CodecKind::SQ8;
p.quantization.train_after = 10000;        // train on a representative sample: this training is final
p.quantization.keep_full_vectors = false;  // after training, keep only the codes
HnswIndex index(768, Metric::Cosine, p);
// ... add vectors ...
index.has_vector_values();                 // false once trained: floats dropped
```

With a codec and `keep_full_vectors = false`, an index keeps full-precision vectors only until its codec is trained; then it drops them and keeps only the codes, one byte per dimension:

- **Memory and files:** about a quarter of full precision for the vectors (in tests a saved index was 3.7 times smaller).
- **Results:** searches compare codes and cannot re-rank (there is nothing left to re-rank with). In tests, recall@10 was about 1.5% below full precision (0.972 to 0.976 against 0.984 to 0.990).
- **Reading back:** `get_vector`, `get` and `scan` return decoded vectors, within half a step of the stored ones; values outside the trained range come back clamped to it.
- **Training is final.** `train_codec()` is refused once only codes are kept: training again could only use decoded vectors and lose more precision (a sample of a few vectors would collapse every code for good). So train on a representative sample: automatically at `train_after` vectors, or with `train_codec()` while the floats are still there. Vectors added later with values outside the trained range are clamped.
- Everything else works from codes: adds, removals, updates, batches, `compact`, and every kind of search. `has_vector_values()` tells whether the floats are still kept.

## Saving and loading

```cpp
index.save("products.hnsw");                        // the whole index: vectors, graph, metadata, settings
auto loaded = HnswIndex::load("products.hnsw");     // a std::unique_ptr<HnswIndex>
auto flat = FlatIndex::load("exact.flat");           // the same for FlatIndex
```

- **Everything is saved:** vectors, the HNSW graph, removed slots and the order they will be reused in, metadata with its schema and payload indexes, HNSW and planner settings, and the random generator's state. A loaded index continues exactly as if the program had never stopped: adding the same vectors afterwards builds the same graph.
- **Saving never damages an existing file.** The index is written to a temporary file next to it (`<path>.tmp-` and a unique suffix) and renamed over `path` only when complete, so a crash or a full disk leaves the previous file intact. (Standard C++ cannot force data onto the physical disk, so a power cut immediately after `save` returns could still lose that latest save.)
- **Loaded files are checked thoroughly.** Each section carries a checksum, every count is checked before memory is allocated, and every link, level and value is validated. A damaged, truncated or foreign file throws `vecdb::FormatError`; a failed load never produces a half-built index.
- **Streams work too:** `save(std::ostream&)` and `load(std::istream&)`, for example with `std::stringstream`. A stream given to `load` must be seekable (files and string streams are).
- A save reads the index, so searches and other saves run alongside it, and changes wait until it finishes. Several threads may save to the **same path** at once: each writes its own temporary file and the last to finish wins, so the file is always complete. (On Windows, two renames over one file at the very same moment can be refused; that save then throws `std::runtime_error` and the file stays complete.)
- Files are portable between operating systems and compilers, and stay readable by future versions of hnsw-lite (the format is versioned; this version writes format 3 and reads 1 to 3). A cosine index saved in format 1, before originals were kept, still loads; `get_vector` then returns its normalized vectors.

## Threads

Every index is safe to use from any number of threads at once, with no lock of your own.

- **Searches** of every kind, **saves**, and lookups (`contains`, `get_metadata`, `size` and so on) run in parallel with each other.
- **Changes** (`add`, `update`, `upsert`, `remove`, `set_metadata`, `create_payload_index`, `set_planner_params`, `compact`) run one at a time: a change waits until the searches running when it arrived have finished, and searches arriving after it wait until it is done. A queued change is never starved by a steady stream of searches.
- **`upsert` is atomic:** the check for an existing vector and the update or insert happen under one lock, so two threads upserting the same ID never collide.
- **Changes cannot be made from inside a search**, for example from a filter predicate, on that index or any other: this throws `std::logic_error`. Waiting would deadlock (the change would wait for the very search that called it), and changes nested inside searches are what create lock cycles between threads. Predicates may still **search** any index, including the one they are filtering.
- **Batch searches** use their own thread pool; predicates in batch searches run on several threads at once, so they must be thread-safe.
- **References** returned by `storage()` and `metadata()` are for inspection while no other thread changes the index: the lock covers the call, not later use of the reference. From concurrent code, use `get_metadata`, `planner_params()` (which returns a copy) and the counting functions.
- **Destroying an index** while another thread still uses it is undefined behavior, as for any C++ object: make sure all searches and changes have returned first.
- Changes run one at a time. To build an HNSW index faster, use [`add_batch`](#adding-many-vectors-at-once), which links vectors on several threads inside a single change.

## Errors and guarantees

| Exception | When |
|---|---|
| `std::invalid_argument` | Wrong dimension; NaN or infinity in a vector, metadata float or radius; an ID already stored; unknown field (strict schema) or wrong type; invalid filter, planner or HNSW settings; an integer above the 64-bit signed range |
| `std::logic_error` | A change (`add`, `update`, `remove`, `set_metadata`, `compact` and the like) attempted while a search is running |
| `vecdb::FormatError` (a `std::runtime_error`) | `load` was given a damaged, truncated or unsupported file, or a file of the other index type |
| `std::runtime_error` | `save` or `load` could not open or write a file |
| `std::bad_alloc` | Out of memory |
| `std::length_error` | More than about 4 billion vectors in one index |

**All-or-nothing:** every change either completes or leaves the index's contents as they were (every stored vector, its metadata, and search results), including when memory runs out partway through. A failed `add` never leaves a half-inserted vector or a new metadata field behind. With HNSW, internal bookkeeping may still move on: if memory runs out while a vector is being linked, its slot is kept free for the next add (so `capacity()` can grow by one), and the random level drawn for it is used up. A `FlatIndex` is restored exactly.

Removing an ID that is not stored is not an error: `remove` returns `false`, as do `set_metadata` and (with an empty result) `get_metadata`.

## Memory

Roughly, each vector costs its floats (4 bytes per dimension, rounded up to a multiple of 64 bytes) plus about 140 bytes for an HNSW graph with `M = 16`, plus its metadata (8 bytes per number, 4 bytes per keyword, 4 bytes per tag). For 1 million 768-dimensional vectors that is about 3.2 GB, almost all of it the vectors themselves. See [Architecture and design](architecture.md#memory-budget) for the details.