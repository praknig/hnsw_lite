# User guide

How to use everything hnsw-lite offers. For building and setup, see [Getting started](getting-started.md); for every class and function, see the [API reference](api-reference.md).

All names below are in the `vecdb` namespace. Code examples assume `using namespace vecdb;`.

## Contents

- [Core ideas](#core-ideas)
- [Creating an index](#creating-an-index)
- [Adding vectors](#adding-vectors)
- [Searching](#searching)
- [Updating vectors](#updating-vectors)
- [Removing vectors and compacting](#removing-vectors-and-compacting)
- [Metadata](#metadata)
- [Filtered search](#filtered-search)
- [Batch search](#batch-search)
- [Range search](#range-search)
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

Every metric follows the same rule: **smaller distance means closer**. With cosine, vectors are normalized automatically when added and when searched, so you do not need to normalize them yourself.

Inner product is not a true distance, and HNSW recall is lower with it on vectors of very different lengths. If you can, normalize your vectors and use cosine.

### HNSW settings

```cpp
HnswParams params;
params.M = 16;                  // links per node; more = better recall, more memory
params.ef_construction = 200;   // effort spent building; more = better graph, slower inserts
params.seed = 42;               // same seed and same operations = identical graph
params.repair_on_remove = true; // repair links around removed vectors
HnswIndex index(768, Metric::Cosine, params);
```

The defaults suit most data. `M` between 8 and 48 is typical; values below 8 leave parts of the graph poorly connected. See [Performance and tuning](performance.md) for measured trade-offs.

## Adding vectors

```cpp
index.add(42, embedding);   // embedding: std::vector<float>, std::array, or any contiguous range of floats
```

- The vector must have exactly the index's dimension.
- NaN and infinite values are rejected.
- An ID can only be stored once at a time. Adding an ID that is already stored throws; use `remove` first to replace a vector.
- `add` copies the vector, so your own buffer can be reused or freed afterwards.

Any failure leaves the index exactly as it was. See [Errors and guarantees](#errors-and-guarantees).

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

## Range search

Return every vector within a distance of the query, instead of a fixed number:

```cpp
auto nearby = flat.search_range(query, l2_radius(0.5f));               // L2: everything within distance 0.5
auto close  = hnsw.search_range(query, 0.2f);                          // cosine: within distance 0.2
auto capped = hnsw.search_range(query, 0.2f, /*max_results=*/100);     // at most the 100 closest
```

- The radius uses the same distance values searches return. For L2 that is the **squared** distance, so convert an ordinary radius with `l2_radius(r)`. For inner product, distances are negative.
- Without `max_results`, there is no limit: a large radius can return the whole index.
- `FlatIndex` is exact. `HnswIndex` is approximate, like its normal search; pass `Strategy::ForceExact` in the options when every vector within the radius must be found.
- Filters and predicates work here too.

## Threads

- **Searches** (all kinds) may run from any number of threads at the same time.
- **Changes** (`add`, `update`, `upsert`, `remove`, `set_metadata`, `create_payload_index`, `set_planner_params`, `compact`) must not run at the same time as anything else, including searches. If your program changes an index from several threads, protect it with a lock, for example a `std::shared_mutex`: shared for searches, exclusive for changes. (Built-in support for concurrent inserts is on the [roadmap](project-notes.md#roadmap).)

## Errors and guarantees

| Exception | When |
|---|---|
| `std::invalid_argument` | Wrong dimension; NaN or infinity in a vector, metadata float or radius; an ID already stored; unknown field (strict schema) or wrong type; invalid filter, planner or HNSW settings; an integer above the 64-bit signed range |
| `std::bad_alloc` | Out of memory |
| `std::length_error` | More than about 4 billion vectors in one index |

**All-or-nothing:** every change either completes or leaves the index exactly as it was, including when memory runs out partway through. A failed `add` never leaves a half-inserted vector or a new metadata field behind.

Removing an ID that is not stored is not an error: `remove` returns `false`, as do `set_metadata` and (with an empty result) `get_metadata`.

## Memory

Roughly, each vector costs its floats (4 bytes per dimension, rounded up to a multiple of 64 bytes) plus about 140 bytes for an HNSW graph with `M = 16`, plus its metadata (8 bytes per number, 4 bytes per keyword, 4 bytes per tag). For 1 million 768-dimensional vectors that is about 3.2 GB, almost all of it the vectors themselves. See [Architecture and design](architecture.md#memory-budget) for the details.