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
- [Changing many vectors by filter](#changing-many-vectors-by-filter)
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

## Changing many vectors by filter

```cpp
using vecdb::Filter;
std::size_t n = index.count_where(Filter::eq("category", "old"));     // preview: how many match
index.remove_where(Filter::eq("category", "old"));                    // like SQL's DELETE ... WHERE
index.set_metadata_where(Filter::lt("year", 2000),                    // like SQL's UPDATE ... WHERE
                         Metadata().set("archived", true).unset("featured"));
```

All three take the same filters as searches and return how many vectors matched.

- **The filter is evaluated once, before anything changes.** So `set_metadata_where` applies to each matching vector exactly once, even when it changes the very field the filter tests (for example renaming a category).
- **A filter naming an unknown field** throws `std::invalid_argument` under a strict schema, and matches nothing under a dynamic one, as in searches.
- **`remove_where` always completes:** once the filter is accepted, every match is removed. If memory runs out while repairing the HNSW graph around them, the repair is cut short (as by `remove`); no match is left behind.
- **`set_metadata_where` is all or nothing:** invalid changes throw `std::invalid_argument`, even when nothing matches, and if memory runs out no vector has changed. It works on a copy of the metadata and swaps it in when complete, so it briefly needs extra memory about the size of the index's metadata.
- **Searches see the index before or after a change, never in between**, and like all changes these cannot be made from inside a search (`std::logic_error`); `count_where` is a read and can.

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
- Files are portable between operating systems and compilers, and stay readable by future versions of hnsw-lite (the format is versioned; this version writes format 2 and reads 1 and 2). A cosine index saved in format 1, before originals were kept, still loads; `get_vector` then returns its normalized vectors.

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