# Developer reference

Every public class, method and parameter of hnsw-lite: what it does, what it expects, what it returns, what can go wrong, and its limits. It is written for developers using the library; for how it works inside, see [Architecture and design](architecture.md), and for a guided tour, the [user guide](user-guide.md).

From Python, see the [Python guide](python.md), which maps every call below to its Python form.

Everything lives in the namespace `vecdb`. Headers are in `include/`; one `#include "collection.h"` brings in what most applications need.

## Contents

1. [Before you start: concepts used everywhere](#1-before-you-start-concepts-used-everywhere)
2. [Collection: the recommended starting point](#2-collection-the-recommended-starting-point)
3. [Metadata, schemas and filters](#3-metadata-schemas-and-filters)
4. [HnswIndex: the graph index](#4-hnswindex-the-graph-index)
5. [FlatIndex: exact search, optionally with an inverted file](#5-flatindex-exact-search-optionally-with-an-inverted-file)
6. [DiskIndex: indexes larger than memory](#6-diskindex-indexes-larger-than-memory)
7. [Sparse vectors, BM25 and hybrid search](#7-sparse-vectors-bm25-and-hybrid-search)
8. [Durability: write-ahead logs and snapshots](#8-durability-write-ahead-logs-and-snapshots)
9. [Health, upkeep, rebuilds and long operations](#9-health-upkeep-rebuilds-and-long-operations)
10. [Errors at a glance](#10-errors-at-a-glance)
11. [Limits at a glance](#11-limits-at-a-glance)
12. [Appendix: complete signatures](#appendix-complete-signatures): every type, field and method with exact parameter types and defaults

---

## 1. Before you start: concepts used everywhere

### Which object to use

| You want | Use |
|---|---|
| Documents with an embedding, text and metadata, hybrid search, crash safety, automatic maintenance | [`Collection`](#2-collection-the-recommended-starting-point) |
| Only vectors (with metadata), fast approximate search, full control over every setting | [`HnswIndex`](#4-hnswindex-the-graph-index) |
| Exact answers, small data, or very fast inserts | [`FlatIndex`](#5-flatindex-exact-search-optionally-with-an-inverted-file) |
| More vectors than fit in memory, built once and searched many times | [`DiskIndex`](#6-diskindex-indexes-larger-than-memory) |
| Keyword or SPLADE vectors on their own | [`SparseIndex`](#7-sparse-vectors-bm25-and-hybrid-search) |

### IDs

Every record has a **user ID**, a `std::uint64_t` of your choosing: any value from 0 to 2⁶⁴−1 works, and IDs need not be consecutive. Within one index or collection an ID is unique: adding an ID that is already stored throws `std::invalid_argument`. Once removed, an ID can be used again. Results always report your IDs, never internal numbers.

**Limitation:** indexes take integer IDs only. Collections can take **string keys** (names, UUIDs) when `CollectionOptions::string_ids` is on: see [expiry and string keys](#28a-expiry-and-string-keys).

### Vectors and dimensions

A vector is a `std::span<const float>`: any `std::vector<float>`, array or pointer-and-length converts to it. Every vector in an index has the same length, the **dimension** (`dim`), fixed when the index is created. Every value must be finite: NaN or infinity throws `std::invalid_argument`, and the index is left unchanged.

Vectors are **copied** into the index; you can reuse or free your buffer as soon as the call returns.

### Metrics and distances

Every search returns `SearchResult { std::uint64_t id; float distance; }`, closest first. **A smaller distance always means closer**, whatever the metric:

| `Metric` | `distance` is | Use it for |
|---|---|---|
| `Metric::L2` | The **squared** Euclidean distance (no square root: faster, and it ranks identically) | Coordinates, or data where actual distances matter |
| `Metric::InnerProduct` | **Minus** the dot product (so a larger dot product is a smaller distance; distances are often negative) | Embeddings trained for dot-product similarity, recommendations |
| `Metric::Cosine` | 1 − cosine similarity: 0 for the same direction, 1 for perpendicular, 2 for opposite | Most text embeddings; the length of the vectors does not matter |

With `Cosine`, vectors and queries are normalized to length 1 automatically; a vector of all zeros stays zero (distance 1 to everything). With `L2`, remember that radii in range searches are squared too: convert an ordinary radius with `l2_radius(r)`, which returns `r * r`.

### Thread safety

Every index and collection is safe to use from many threads at once, with no locking of your own:

- **Reads** (searches, lookups, saves, health reports) run in parallel with each other.
- **Changes** (adds, updates, removals, metadata changes) run one at a time, and each waits for running reads to finish. While a change waits, new reads wait too, so a steady stream of searches can never starve a change.
- **No change from inside an index operation on the same thread.** If a predicate (or a progress callback) running inside an index operation tries to change *any* index or collection, the change throws `std::logic_error` instead of deadlocking. Reading from inside a predicate is fine.

### Errors and "all or nothing"

The library reports problems with exceptions (see [Errors at a glance](#10-errors-at-a-glance)). Every change is **all or nothing**: if it throws, for any reason including running out of memory (`std::bad_alloc`), the index or collection is exactly as it was before the call. Where an operation documents a weaker guarantee, the method's description says so.

---

## 2. Collection: the recommended starting point

`#include "collection.h"`

A `Collection` stores **documents**. Each document has a user ID and, all optional:

- a **dense vector** (an embedding), searched with an HNSW graph (the *dense side*);
- **text**, searched with BM25 keyword scoring (the *text side*);
- a **learned sparse vector** such as SPLADE output (the *sparse side*);
- **metadata** (typed fields: numbers, booleans, keywords, tag sets), used by filters.

One search can combine a vector, text and a sparse vector; their rankings are fused into one. A collection can be **durable** (crash-safe, in a directory) and **maintains itself** (repairs the graph, retrains compression, takes snapshots).

```cpp
CollectionOptions options = CollectionOptions::preset(768, Preset::Balanced, 1'000'000);
auto docs = Collection::open("docs.db", options);       // durable; or Collection docs(options) in memory

Document d;
d.vector = embedding;                                    // std::vector<float> of 768 values
d.text = "the document text";
d.metadata = Metadata().set("lang", "en").set("year", 2024);
docs->upsert(42, d);

CollectionQuery q;
q.vector = query_embedding;
q.text = "what the user typed";
CollectionSearchOptions so;
so.filter = Filter::eq("lang", "en");
for (const CollectionHit& hit : docs->search(q, 10, so)) { /* hit.id, hit.score */ }
```

### 2.1 `CollectionOptions`

Settings chosen when a collection is created. Start from [`CollectionOptions::preset`](#22-presets) and adjust what you need.

| Field | Default | What it does |
|---|---|---|
| `dim` | 0 | Length of the dense vectors. **0 means no dense side**: the collection holds only text and/or sparse vectors. |
| `metric` | `Metric::Cosine` | Distance of the dense side (see [Metrics](#metrics-and-distances)). |
| `hnsw` | `HnswParams{}` | Settings of the dense side's graph and compression: see [`HnswParams`](#41-hnswparams) and [`QuantizationParams`](#42-quantizationparams). |
| `schema` | `Schema::dynamic()` | Which metadata fields are allowed: see [Schemas](#33-schema). |
| `text` | `true` | Whether the collection has a text side (BM25). |
| `bm25` | `Bm25Params{}` | BM25 settings: `k1` (1.2), `b` (0.75), `avg_doc_length` (100 tokens). **Set `avg_doc_length` near your documents' mean length in words**: it is fixed when the collection is created and used for every document. See [`Bm25Params`](#74-bm25). |
| `sparse` | `false` | Whether the collection has a learned sparse side (for SPLADE and similar vectors). |
| `store_text` | `true` | Keep each document's text, so `get()` and search results can return it. `false` saves memory; the text is still indexed for search. |
| `durability` | `DurableOptions{}` | For `Collection::open` only: snapshots kept and when the log is flushed to disk. See [`DurableOptions`](#81-durableoptions-and-durability). |
| `maintenance` | `MaintenancePolicy{}` | When maintenance runs (manual by default). See [Maintenance](#28-maintenance). **Not saved** with the collection: `load()` starts with manual maintenance, and `open()` takes it from the options you pass. |
| `ttl` | off | **Opt-in expiry** (`TtlOptions { bool enabled; std::int64_t default_ms; }`): documents with an expiry time (`Document::expires_at`, or now + `default_ms` when that is above 0) disappear from searches, counts, `get` and `contains` the moment it passes, and are freed by `remove_expired()` or maintenance. **Saved with the collection** (only when not the defaults, so other files are unchanged): `load()` and `open()` keep it. Passing it at `open()` switches the feature on; leaving it out never switches it off (`set_ttl` and `set_string_ids` do). |
| `string_ids` | false | **Opt-in string keys** (names, UUIDs): enables the `std::string_view` overloads of `add`, `upsert`, `get`, `contains` and `remove`. **Saved with the collection** (only when not the defaults, so other files are unchanged): `load()` and `open()` keep it. Passing it at `open()` switches the feature on; leaving it out never switches it off (`set_ttl` and `set_string_ids` do). |
| `ef` | 0 | Default beam width of dense searches (higher: better recall, slower). 0 uses the library default, max(candidates, 64). **Not saved**, like `maintenance`. A search's own `CollectionSearchOptions::ef` overrides it. |

At least one side must exist (`dim > 0`, `text` or `sparse`); otherwise the constructor throws `std::invalid_argument`.

**Text and sparse vectors live on separate sides** because their term numbers mean different things: BM25 terms are hashes of words, while SPLADE terms are positions in a model's vocabulary. Mixing them in one index would make unrelated terms collide.

### 2.2 Presets

```cpp
static CollectionOptions CollectionOptions::preset(std::size_t dim, Preset preset = Preset::Balanced,
                                                   std::size_t expected_size = 100000);
```

Returns ready-made options for `dim`-dimensional vectors and about `expected_size` documents. Every preset sets cosine distance, a text side, and **background maintenance**; they differ in the graph, compression and search beam:

| `Preset` | Graph | Compression | Default search beam (`ef`) | Measured recall@10 | Size of the dense side |
|---|---|---|---|---|---|
| `Balanced` | `M` 16, `ef_construction` 200; from a million documents `M` 24, `ef_construction` 300 | none | 128 | 0.982 | 2.5 MB |
| `HighRecall` | `M` 32, `ef_construction` 400; from a million `M` 48, `ef_construction` 500 | none | 256 | 1.000 | 2.8 MB |
| `LowMemory` | `M` 12, `ef_construction` 150 | SQ8, codes only (one byte per dimension) | 128 | 0.933 | 0.63 MB |

The measurements are for 4,000 clustered 64-dimensional vectors, with recall measured against an exact search; your data will differ, so measure on it (for example with [`HnswIndex::health`](#47-health-upkeep-compaction-and-rebuilds)).

- `dim` may be 0, for a text-only collection.
- `expected_size` only matters across the one-million threshold: larger graphs need more links to keep recall up.
- **Adjust anything afterwards:** the result is an ordinary `CollectionOptions`.
- Throws `std::invalid_argument` for a value that is not one of the three presets.

**Limitations:** `LowMemory` keeps no full-precision vectors, so `get()` returns decoded vectors (close to, but not exactly, what you stored), and its compression cannot be retrained later (see [`keep_full_vectors`](#42-quantizationparams)). Product quantization is deliberately not used by any preset: without full vectors to re-rank against, its recall measured only 0.3 to 0.4.

### 2.3 Creating and opening

```cpp
explicit Collection(CollectionOptions options);
static std::unique_ptr<Collection> Collection::open(const std::string& dir, CollectionOptions options,
                                                    RecoveryStats* stats = nullptr);
```

- **`Collection(options)`** creates an empty collection **in memory**. Nothing is written to disk unless you call `save()`. Throws `std::invalid_argument` for invalid options (no side at all, invalid HNSW, BM25 or maintenance settings). If `options.maintenance.mode` is `Background`, the maintenance thread starts immediately.
- **`Collection::open(dir, options, stats)`** opens a **durable** collection in directory `dir`:
  - If `dir` does not exist or holds no collection, it is **created** (with an empty snapshot).
  - Otherwise the collection is **recovered**: the newest snapshot is loaded and every change logged since is replayed, so every change that returned before a crash is back (subject to the [durability mode](#81-durableoptions-and-durability)).
  - When recovering, `dim`, `metric`, `text` and `sparse` **must match** what the directory holds (`std::invalid_argument` otherwise). Other stored settings (HNSW, schema, BM25, `store_text`) come from the directory; `durability`, `maintenance` and `ef` come from the options you pass.
  - `stats`, if given, receives what recovery did: see [`RecoveryStats`](#82-recoverystats).
  - Only **one process** may have a directory open: a second `open` throws `std::runtime_error` (the directory holds a lock file while open). The lock is released when the collection is destroyed, or if the process dies.
  - Throws `FormatError` if the directory cannot be recovered (damage that a crash cannot cause: see [Durability](#8-durability-write-ahead-logs-and-snapshots)).

The collection is closed by destroying it (`docs.reset()`, or leaving its scope): the background maintenance thread stops, the log is flushed, and the lock is released.

### 2.4 Documents

```cpp
struct Document {
    std::vector<float> vector;   // dense vector (empty: none)
    std::string text;            // text for BM25 (empty: none)
    SparseVector sparse;         // learned sparse vector (empty: none)
    Metadata metadata;
};
```

`Document` also has `std::optional<std::int64_t> expires_at` (milliseconds since 1970; with `ttl` enabled only, otherwise `std::invalid_argument`).

Every part is optional. A document goes into each side for which it has a part: a document with only text is found by text queries but not by vector queries. A document with **no parts at all** is still stored (with its metadata) and found by filters (`count_where`, `remove_where`), but never by `search`.

Rules, checked **before anything changes** (a violation throws `std::invalid_argument`):

- A part for a side the collection does not have is an error (for example `vector` when `dim` is 0).
- `vector` must have exactly `dim` finite values.
- `sparse` must have as many `values` as `indices`, distinct indices, and finite values (see [`SparseVector`](#71-sparsevector)).
- `metadata` must fit the schema.
- Text with no words (such as `"!!!"`) is accepted and stored, but matches no query.

### 2.5 Changing documents

| Method | What it does | Returns |
|---|---|---|
| `void add(std::uint64_t id, const Document& document)` | Adds a new document. Throws `std::invalid_argument` if `id` is already stored, or the document is invalid. | |
| `bool upsert(std::uint64_t id, const Document& document)` | Adds the document, or **replaces the whole document** if `id` exists. A part left out of the new version is **removed** from its side (for example, upserting without text removes the old text). Metadata is replaced too. | `true` if added, `false` if replaced |
| `bool remove(std::uint64_t id)` | Removes the document from every side. | `false` if `id` was not stored |
| `std::size_t remove_batch(std::span<const std::uint64_t> ids)` | Removes every listed ID that is stored; unknown and repeated IDs are skipped. | how many were removed |
| `std::size_t remove_where(const Filter& filter)` | Removes every document whose metadata matches `filter` (like SQL's `DELETE WHERE`). | how many were removed |
| `bool set_metadata(std::uint64_t id, const Metadata& metadata)` | Changes **only the given fields** of one document's metadata; `Metadata().unset("field")` clears a field. Vectors and text are untouched. | `false` if `id` was not stored |
| `std::size_t set_metadata_where(const Filter& filter, const Metadata& changes)` | Applies `changes` to every document matching `filter`, all or nothing. | how many changed |

**Guarantees.** Every change is all or nothing: the document is checked first, then the change is logged (if durable), then applied side by side; if a later side runs out of memory, the sides already changed are restored. Searches see a change completely or not at all.

**Limitations:**
- Changes run **one at a time** (searches run in parallel with each other, and wait for the running change).
- `remove_where` finds the matches first and removes them under a separate exclusive lock: a matching document added in between stays, as if added afterwards.
- If restoring after an out-of-memory failure *also* fails (a second failure in the same change), the sides may disagree; the collection then refuses further changes (`std::runtime_error`) until reopened. This needs two allocation failures in one change.
- Calling a change from inside a search's predicate on the same collection throws `std::logic_error`.

### 2.6 Reading documents

| Method | What it does |
|---|---|
| `bool contains(std::uint64_t id) const` | Whether `id` is stored. |
| `std::size_t size() const` | Number of documents. |
| `std::size_t count_where(const Filter& filter) const` | How many documents match `filter`: a preview of `remove_where`. |
| `std::optional<DocumentRecord> get(std::uint64_t id) const` | The document, or nothing if `id` is not stored. |
| `CollectionPage scan(const CollectionScanOptions& options = {}) const` | Lists documents **a page at a time, in ascending ID order**, optionally only those matching a filter (see *Reading every document* below). Throws `std::invalid_argument` if `limit` is 0, or for an unknown field under a strict schema. |
| `std::vector<CollectionGroup> search_groups(const CollectionQuery& query, const GroupBy& group_by, const CollectionSearchOptions& options = {}) const` | **Group-by:** the `groups` best values of `group_by.field` (keyword, integer or boolean), each with its `per_group` best documents, ranked as `search()` ranks them. Documents without the field are in no group. Widens the search 4× at a time until the groups are found and filled, every match was seen, or `max_candidates` (0: all) is reached: with an exact search, equal to grouping the full ranking. Under one read lock. `GroupBy { field, groups = 10, per_group = 3, max_candidates = 0 }`; `CollectionGroup { Value value; std::vector<CollectionHit> hits; }`. Other field types, an empty name, or an unknown field under a strict schema throw `std::invalid_argument`. |
| `std::vector<CollectionHit> recommend(const Recommend& examples, std::size_t k, const CollectionSearchOptions& options = {}) const` | **Recommendation:** documents like `examples.positive` and unlike `examples.negative` (`Recommend { std::vector<std::uint64_t> positive, negative; }`): a vector search from avg(positive), or 2 avg(positive) - avg(negative), each example normalized first for cosine; computed in double and clamped to float's range. The examples are never returned; options apply as in `search()`. Under one read lock. Throws `std::invalid_argument` without a positive example, for an unknown or expired document, one without a vector, one in both lists, or a collection without vectors. |
| `std::vector<FacetResult> facets(requests, filter)`, `facets_for(ids, requests)` | Faceted counts, as for `HnswIndex` (see *Faceted search*); expired documents are never counted. |
| `void create_payload_index(std::string_view field)` | Counts and posting lists for a keyword, boolean or tag field (as for `HnswIndex`); logged for durable collections. |
| `Schema schema() const`, `std::vector<std::string> payload_indexes() const` | The schema in force (also in `options().schema`, which always reports the current one) and the indexed fields. |
| `CollectionOptions options() const` | A copy of the collection's settings, taken under its lock, since `set_ttl` and `set_string_ids` may change them from another thread (for a recovered collection, the stored ones). |
| `dense_index()`, `text_index()`, `sparse_index()` | Read-only pointers to the sides (null if absent), for example to call `health()` on the dense side. Do not keep them beyond the collection's lifetime. |

```cpp
struct DocumentRecord {
    std::uint64_t id;
    std::optional<std::vector<float>> vector;   // absent if the document has no vector
    std::string text;                           // empty unless store_text (or if it had none)
    std::optional<SparseVector> sparse;         // learned sparse vector, if any
    Metadata metadata;
};
```

`DocumentRecord` also reports `key` (with string keys) and `expires_at` (with expiry), taken from reserved metadata fields, which are **not** included in `metadata`.

`vector` is the vector as you gave it, except when only codes are kept (the `LowMemory` preset, or half precision with `keep_full_vectors = false`), where it is decoded from the codes and therefore approximate. The **BM25 terms of the text are not returned** (they are derived from the text).

**Reading every document.** `scan()` returns a page of documents and the cursor to the next:

```cpp
struct CollectionScanOptions {
    std::optional<Filter> filter;        // only documents matching it (all if empty)
    std::size_t limit = 100;             // documents per page, at least 1
    std::optional<std::uint64_t> after;  // continue after this ID: the previous page's `next`
    bool with_vectors = true;            // the dense and sparse vectors
    bool with_text = true;
    bool with_metadata = true;           // keys and expiry times come back either way
};
struct CollectionPage {
    std::vector<DocumentRecord> documents;
    std::optional<std::uint64_t> next;   // empty on the last page
};

CollectionScanOptions o;
o.limit = 1000;
for (;;) {
    const CollectionPage page = docs.scan(o);
    for (const DocumentRecord& d : page.documents) { /* ... */ }
    if (!page.next) break;
    o.after = page.next;
}
```

Documents come in ascending ID order. Expired documents are left out, as everywhere. Each page is read under one lock, so it is consistent, and changes between pages never repeat or skip a document that stays stored: a document removed ahead of the cursor is not listed, one added ahead of it is, and one added behind it is not. A document in a page is the same as `get()` returns for it. Each page costs one pass over the collection, so read a large collection in large pages (thousands of documents).

**What a file holds** (`inspect.h`). `StoredInfo identify(const std::string& path)` reads only headers: `kind` (`StoredKind::Flat`, `Hnsw`, `Sparse`, `Collection` or `Disk`; `stored_kind_name()` gives its name), `durable`, the format `version`, `bytes`, and for a durable directory its `identity` and the LSNs of its `snapshots` and log `segments`. Throws `FormatError` for anything else, `std::runtime_error` if the path cannot be read.

### 2.7 Searching

```cpp
std::vector<CollectionHit> search(const CollectionQuery& query, std::size_t k,
                                  const CollectionSearchOptions& options = {}) const;
```

Returns up to `k` documents, best first.

```cpp
struct CollectionQuery {
    std::vector<float> vector;   // dense query (empty: none)
    std::string text;            // keyword query, scored with BM25 (empty: none)
    SparseVector sparse;         // learned sparse query (empty: none)
};
```

Give **at least one** part (`std::invalid_argument` otherwise). A part for a side the collection lacks is an error. With **one** part, results follow that side's own ranking. With **several**, each side returns its best candidates and the lists are **fused** into one ranking.

```cpp
struct CollectionSearchOptions {
    std::optional<Filter> filter;
    std::function<bool(std::uint64_t)> predicate;
    Fusion fusion = Fusion::Rrf;
    double rrf_k = 60.0;
    double vector_weight = 1.0, text_weight = 1.0, sparse_weight = 1.0;
    std::size_t candidates = 0;
    std::size_t ef = 0;
    bool with_metadata = false;
    bool with_text = false;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    bool* stopped_early = nullptr;
};
```

| Field | What it does |
|---|---|
| `filter` | Only documents whose metadata matches are returned, from every side. See [Filters](#34-filter). |
| `predicate` | Any condition on the ID, for data outside the collection (permissions, say). Both `filter` and `predicate` must pass. It may read the collection (`get`, `contains`, even a nested search) but **must not change** it or any other index (`std::logic_error`). |
| `fusion` | `Fusion::Rrf` (default): **Reciprocal Rank Fusion**. Each document scores `weight / (rrf_k + rank)` summed over the lists that found it (rank 1 is best). It uses ranks only, so it needs no tuning and is robust when the sides' scores are on different scales. `Fusion::Weighted`: each list's scores are scaled to [0, 1] (best 1, worst 0) and added with the weights; it keeps *how much* better one result is than the next, but is more sensitive to outliers. |
| `rrf_k` | The RRF constant (60, the value from the original paper). Smaller values favor the top ranks more strongly. Must be finite and non-negative. |
| `vector_weight`, `text_weight`, `sparse_weight` | How much each side counts in the fusion. 0 ignores a side's ranking (its candidates still appear if another side found them). Finite and non-negative. |
| `candidates` | How many candidates each side contributes before fusion; 0 means max(4·k, 20). More candidates give fusion more to work with, at the cost of more work per side. |
| `ef` | Beam width of the dense search. 0 uses `CollectionOptions::ef`, or the library default max(candidates, 64). |
| `with_metadata`, `with_text` | Fill `metadata` and `text` in each hit (copying them costs time; off by default). |
| `modifiers` | Opt-in re-scoring by metadata (recency decay, confidence, boosts): see [score modifiers](#46-searching). Hits then carry `base_score`. |
| `diversity` | Opt-in **maximal marginal relevance** (MMR), λ in [0, 1]: each next hit maximizes λ × relevance − (1 − λ) × its highest cosine similarity to the hits already chosen. 1 keeps the ranking; around 0.3 to 0.7 trades relevance for variety. Works on the fused candidates (each side's `candidates`, default max(4k, 20)): it can only choose among them. Needs a dense side. |
| `deadline`, `cancel` | Stop every side when this moment passes, or when another thread sets the flag to `true`. The hits found so far are returned, fused and sorted; nothing is thrown. |
| `stopped_early` | If set, receives `true` when a deadline or cancellation stopped any side (results may then be incomplete), `false` otherwise. |

```cpp
struct CollectionHit {
    std::uint64_t id;
    double score;                                       // fused score: higher is better
    std::optional<std::size_t> vector_rank, text_rank, sparse_rank;   // 1-based, if that side found it
    std::optional<float> vector_distance;               // dense distance (smaller is closer)
    std::optional<float> text_score, sparse_score;      // dot products (larger is better)
    std::optional<Metadata> metadata;                   // if with_metadata
    std::optional<std::string> text;                    // if with_text
};
```

Each hit says **which sides found it**, at what rank, and with what raw distance or score, so you can explain or re-rank results. Ties in `score` are broken by smaller ID.

**Limitations:**
- Searching an empty collection, or with `k = 0`, returns no hits.
- Text queries use the same BM25 tokenizer as documents: lowercase runs of letters and digits, **no stemming** ("apples" does not match "apple") and no stop-word removal. Tokenize your way and use the sparse side if you need either.
- The dense side is approximate (HNSW): a document can occasionally be missed. Raise `ef` for better recall.

### 2.8 Maintenance

A collection that changes over time needs care: replaced and removed vectors leave the HNSW graph less well connected (*churn*), compression trained on old data drifts as data moves, and a durable collection's log grows until a snapshot is taken. **Maintenance** handles all three.

```cpp
struct MaintenancePolicy {
    Maintenance mode = Maintenance::Manual;            // or Maintenance::Background
    std::chrono::milliseconds interval{1000};
    std::chrono::milliseconds budget{250};
    std::uint64_t checkpoint_bytes = 64ull << 20;      // 64 MiB
    std::uint64_t checkpoint_records = 100000;
    std::chrono::seconds checkpoint_age{600};
    double health_check_churn = 0.05;
    double upkeep_churn = 0.2;
    std::size_t upkeep_nodes = 0;
    double rebuild_check_churn = 0.5;
    double rebuild_drift = 4.0;
};
```

**Churn** is the number of dense vectors **replaced or removed** since the work last ran, divided by the number of live vectors. Additions are not churn: they do not damage the graph.

| Field | What it does |
|---|---|
| `mode` | `Manual` (default): nothing runs unless you call `maintain()`. `Background`: a background thread runs `maintain(budget)` every `interval`. Presets choose `Background`. |
| `interval` | Background mode: time between rounds (at least 1 ms). |
| `budget` | Background mode: time per round for health checks and upkeep (0: no limit). |
| `checkpoint_bytes`, `checkpoint_records` | Durable collections: take a snapshot (`checkpoint()`) once the log since the last one reaches this size or this many records. Snapshots keep the log, and so recovery time, bounded. |
| `checkpoint_age` | And whenever a logged change has waited this long without a snapshot. |
| `health_check_churn` | Past this churn, run a quick health check (reachability only) and run upkeep if it finds unreachable vectors, links to removed vectors or stale links. |
| `upkeep_churn` | Past this churn, run upkeep regardless of health. |
| `upkeep_nodes` | Most vectors repaired per upkeep call (0: no limit; the time budget still applies). |
| `rebuild_check_churn` | Collections with compression: past this churn, measure the codec's error against a retrained one. |
| `rebuild_drift` | Rebuild (retraining the codec) when its error exceeds this many times the retrained error. |

Invalid values (an interval under 1 ms, negative or non-finite thresholds) throw `std::invalid_argument`.

| Method | What it does |
|---|---|
| `MaintenanceReport maintain(std::chrono::milliseconds budget = 0)` | Runs **only what is due**, in this order: a health check and upkeep of the dense side (within `budget`; 0 means no limit), a rebuild if the codec drifted, then a checkpoint if the log has grown. Searches and changes continue meanwhile. Returns what it did. |
| `void set_maintenance(const MaintenancePolicy& policy)` | Changes the policy at any time; starts or stops the background thread as needed. |
| `std::optional<MaintenanceReport> last_maintenance() const` | The last round's report (manual or background), or nothing if none ran. |
| `MaintenanceStatus maintenance_status() const` | The counters maintenance decides from, right now. |

```cpp
struct MaintenanceReport {
    std::chrono::system_clock::time_point when;
    std::chrono::microseconds elapsed;
    std::optional<HealthReport> health;   // if a health check ran
    bool upkept; UpkeepStats upkeep;      // upkeep's statistics
    bool rebuilt; RebuildStats rebuild;
    bool checkpointed; std::uint64_t checkpoint_lsn;
    bool budget_exhausted;                // due work was left for the next round
    std::vector<std::string> notes;       // each decision in words, for example "upkeep due: churn 0.214 reached 0.200"
    std::string error;                    // background rounds: what stopped this one (empty if none)
};

struct MaintenanceStatus {
    std::uint64_t log_bytes, log_records;   // log since the last snapshot (durable collections; 0 otherwise)
    std::chrono::seconds since_checkpoint;
    double churn_since_upkeep, churn_since_rebuild;
    std::size_t rounds;                     // rounds run so far
};
```

**How budgets work.** Upkeep stops when the budget runs out, but only after it has repaired **at least one vector** in that round, so repeated rounds always finish the work. A **rebuild**, once started, runs to the end (it does not block searches or changes, but it can take as long as building the index again). A **checkpoint** always runs when due, whatever the budget, because it is what keeps the log bounded.

**Limitations:**
- Maintenance is **not logged**: its effect on the graph is kept by the next snapshot. After a crash, a recovered collection has the same documents, but its graph may be as it was before the last maintenance round.
- In background mode, an error in a round (a full disk during a checkpoint, say) is recorded in that round's report (`error`) and the next round tries again; nothing is thrown to your code.
- `maintain()` and `set_maintenance()` called from inside a search of the same collection throw `std::logic_error`.
- The text and sparse sides need no maintenance: they compact themselves.

### 2.8a Expiry and string keys

| Method | What it does |
|---|---|
| `std::size_t remove_expired()` | With `ttl`: removes every expired document, returning how many (they are already hidden). `maintain()` calls it and notes "removed N expired documents". 0 without `ttl`. |
| `static std::uint64_t id_of(std::string_view key)` | A key's numeric id: a 64-bit hash (FNV-1a with a SplitMix64 finish). |
| `bool set_metadata(std::string_view key, const Metadata& metadata)` | Changes only the given fields of the document stored under that exact key; `false` if there is none (absent, expired, or another key's document). |
| `void set_ttl(const TtlOptions& ttl)`, `void set_string_ids(bool on)` | The runtime settings, at any time, from any thread (an operation already running uses the settings it started with): **changes are saved with the collection** and, for durable collections, logged at once (they survive a crash before the next checkpoint). Throw `std::invalid_argument` if a strict schema lacks the reserved field; `open()` checks the same. |
| `add(key, doc)`, `upsert(key, doc)`, `get(key)`, `contains(key)`, `remove(key)` | With `string_ids` (`std::logic_error` otherwise): the document's id is `id_of(key)`, and the key is kept with it. `add` refuses a key already present; both refuse a key whose id belongs to **another** key (`std::invalid_argument`), so two keys are never mixed up. `get`, `contains` and `remove` never act on a document that was not stored under that exact key. Hits report `key`. |

**Reserved metadata names:** `_expires_at` (an integer) and `_key` (a keyword) hold the expiry time and the key; your metadata may not use them (`std::invalid_argument`). Strict schemas gain them automatically when the features are on at creation; a strict schema created without them cannot later store them. Expiry times are fixed when a document is written and logged with it, so recovery after a crash keeps them exactly.

**Limitations:** `size()` counts expired documents until they are removed. Hash collisions between keys are refused rather than resolved (about one chance in 3 × 10⁶ at ten million keys); refused keys can be stored under different text.

### 2.9 Durability and files

| Method | What it does |
|---|---|
| `std::uint64_t checkpoint()` | Durable collections: starts a new log segment and writes a snapshot as of that moment (searches continue; changes wait while it is written), then deletes log segments and snapshots no longer needed. Returns the snapshot's sequence number. Throws `std::logic_error` for an in-memory collection, or from inside a search of the same collection. Background maintenance calls it for you. |
| `bool durable() const` | Whether the collection was opened with `open()`. |
| `void save(const std::string& path) const`, `void save(std::ostream& out) const` | Writes the whole collection to one file (for in-memory collections; a durable one saves itself). The file is written to a temporary name and renamed into place, so an existing file is never damaged by a failed save. Throws `std::runtime_error` if it cannot be written. |
| `static std::unique_ptr<Collection> load(const std::string& path)`, `load(std::istream& in)` | Reads a file written by `save()`. Throws `FormatError` for a damaged file, a file of another kind (an index file, for example), or content that does not fit together; `std::runtime_error` if it cannot be opened. The loaded collection starts with **manual** maintenance. |
| `bool replay(std::span<const WalRecord> records)` | Advanced: applies logged records as recovery does. See [Durability](#8-durability-write-ahead-logs-and-snapshots). |

**Limitations:** `save()` does not force data onto the physical disk (a power cut right after it returns could lose the file); durable collections do, according to their durability mode. One process per durable directory.

---

## 3. Metadata, schemas and filters

`#include "metadata.h"` and `"filter.h"` (both included by every index header)

### 3.1 Field types and values

Each metadata field has one of five types (`FieldType`):

| `FieldType` | Holds | C++ values that produce it |
|---|---|---|
| `Int` | A 64-bit signed integer | any integer type except `bool` (unsigned values above the `int64_t` range throw `std::invalid_argument`) |
| `Float` | A 64-bit floating-point number | `float`, `double` (NaN and infinity are rejected) |
| `Bool` | `true` or `false` | `bool` |
| `Keyword` | A string, compared exactly (no case folding, no tokenizing) | `std::string`, `const char*`, anything convertible to a string |
| `Tags` | A set of strings | `std::vector<std::string>`, through `set_tags` |

Integers and floats **compare exactly with each other** in filters: `Filter::eq("year", 2024.0)` matches an integer field holding 2024.

### 3.2 `Metadata`

A small map from field names to values, attached to a record.

| Method | What it does |
|---|---|
| `Metadata& set(std::string name, T&& value)` | Sets a field; its type comes from the C++ type of `value` (table above). Returns the `Metadata`, so calls chain: `Metadata().set("a", 1).set("b", "x")`. |
| `Metadata& set_tags(std::string name, std::vector<std::string> tags)` | Sets a tag-set field. It behaves as a set: order and duplicates do not matter to filters or comparisons. |
| `Metadata& unset(std::string name)` | Marks a field to be **cleared**. Only meaningful for `set_metadata()` and `set_metadata_where()`, which change only the fields they are given: this is how you remove one. In a new record it simply means "no value". |
| `std::optional<Value> get(std::string_view name) const` | The value of a field, or nothing if absent (or marked unset). A `Value` has `type` and the member for that type: `i` (`Int`), `f` (`Float`), `b` (`Bool`), `s` (`Keyword`), `tags` (`Tags`). |
| `const std::map<std::string, std::optional<Value>>& fields() const` | Every field set on this object (`nullopt` for unset ones). |
| `bool empty() const`, `std::size_t size() const` | Whether any field is set; how many fields are set (including ones marked unset). |
| `operator==` | Same fields with the same values (tag sets compared as sets). |

To build a `Value` directly (rarely needed: `set` and the filter functions build them for you), use `Value::of_int(std::int64_t)`, `Value::of_float(double)`, `Value::of_bool(bool)`, `Value::of_keyword(std::string)`, `Value::of_tags(std::vector<std::string>)`, or `Value::from(x)`, which picks the type from `x`'s C++ type as in the table above. A `Value`'s fields are `type`, `i`, `f`, `b`, `s` and `tags`; only the one matching `type` is meaningful.

**Limitations:** strings are stored once per distinct value per index (repeated keywords cost little), and strings no record uses any more are cleaned up automatically, so a field with ever-new values (session IDs) does not grow memory forever; very large strings and huge tag sets still cost memory like any data. There is no date or geographic type: store dates as integers (for example seconds since 1970) and compare them with ranges.

### 3.3 `Schema`

Decides which fields a record may carry. Passed when an index or collection is created.

| Constructor | Behavior |
|---|---|
| `Schema::dynamic(std::vector<FieldSpec> fields = {})` (the default) | The listed fields, plus **any new field**, created on first use with the type of its first value. A filter naming a field that does not exist yet **matches nothing** (it may simply not exist yet). |
| `Schema::strict(std::vector<FieldSpec> fields)` | **Only** the listed fields. A record with any other field is rejected (`std::invalid_argument`), and so is a filter naming one. Use it to catch typos early. |

A `FieldSpec` is `{ std::string name; FieldType type; }`. In both modes **a field keeps one type**: once `year` is an integer, storing a string in it throws `std::invalid_argument`. Empty or duplicate field names throw `std::invalid_argument`.

### 3.3a Dates

Dates are stored as **integers: milliseconds since 1970-01-01T00:00:00Z** (UTC), in `Int` fields, so they filter like any number and need no new file format.

| Function | What it does |
|---|---|
| `std::int64_t parse_date(std::string_view iso)` | ISO 8601: `"2024-05-17"`, `"2024-05-17T09:30"`, `"2024-05-17T09:30:15.250"`, optionally ending `Z` or an offset like `+02:00` (without one, UTC). Validates the calendar (leap years included); anything else throws `std::invalid_argument`. |
| `std::string format_date(std::int64_t ms)` | `"YYYY-MM-DDTHH:MM:SS.mmmZ"`; `parse_date(format_date(ms)) == ms` |
| `std::int64_t now_ms()` | The current time |
| `Metadata& Metadata::set_date(std::string name, std::string_view iso)` | `set(name, parse_date(iso))` |

```cpp
Filter recent = Filter::ge("published", parse_date("2024-01-01"));
```

**Limitation:** a date reads back as an integer (format it with `format_date`); there is no separate date type in schemas.

### 3.4 `Filter`

A condition on metadata, built with these functions and combined with `&&`, `||` and `!`:

| Function | Matches records where |
|---|---|
| `Filter::all()` | always (the same as no filter) |
| `Filter::eq(field, v)`, `ne`, `lt`, `le`, `gt`, `ge` | the field compares to `v` that way |
| `Filter::between(field, lo, hi)` | lo ≤ field ≤ hi (inclusive; `lo > hi` matches nothing) |
| `Filter::in(field, {v1, v2, ...})` | the field equals any of the values (an empty list matches nothing) |
| `Filter::has_tag(field, tag)` | the tag set contains `tag` |
| `Filter::has_any_tag(field, {tags})` | the tag set contains at least one of them |
| `Filter::has_all_tags(field, {tags})` | the tag set contains all of them (with no tags: every record that has the field) |
| `Filter::exists(field)` | the record has a value for the field |
| `Filter::geo_radius(lat_field, lon_field, lat, lon, meters)` | within `meters` of (lat, lon) by great-circle distance (haversine, mean Earth radius 6,371,008.8 m); the two fields numeric, in degrees. Missing or out-of-range coordinates never match; an out-of-range center, or a negative or non-finite radius, throws `std::invalid_argument` |
| `Filter::geo_box(lat_field, lon_field, south, west, north, east)` | latitude in [south, north] and longitude in [west, east], edges included; west > east crosses the antimeridian. Throws for out-of-range coordinates or south > north |
| `a && b`, `a \|\| b`, `!a` | both, either, not |

```cpp
Filter f = Filter::eq("lang", "en") && (Filter::ge("year", 2020) || Filter::has_tag("tags", "featured"));
```

Rules worth knowing:

- **A comparison never matches a record that lacks the field**, including `ne` (as in SQL): `Filter::ne("lang", "en")` does not match records without `lang`. `!` negates the whole condition, so `!Filter::eq("lang", "en")` *does* match them.
- Keywords support `eq`, `ne` and `in`; booleans `eq`, `ne` and `in`; numbers every comparison; tag sets only the tag functions. Using another combination throws `std::invalid_argument` when the filter is used.
- A filter is checked against the index's schema **each time it is used**: an unknown field throws under a strict schema and matches nothing under a dynamic one; a wrong value type throws.
- Filters are cheap values: copy and reuse them freely, across indexes too. `bool is_all() const` tells whether a filter is the default "match everything" one.

**Limitations:** filters nest at most 256 levels deep (`Filter::kMaxDepth`); chaining thousands of `&&` or `||` in a loop throws. Combine many values with `in()` or `has_any_tag()` instead. Filters compare whole values only: there is no substring, prefix or full-text matching on keywords (use the text side of a collection for words).

---

## 4. HnswIndex: the graph index

`#include "hnsw_index.h"`

An approximate nearest-neighbor index based on HNSW (Hierarchical Navigable Small World graphs). It stores vectors with optional metadata, finds the closest ones in a fraction of the time an exact scan takes, and supports updates, real deletion, filters, compression, persistence and durability.

### 4.1 `HnswParams`

| Field | Default | What it does |
|---|---|---|
| `M` | 16 | Links per node (2·M on the bottom level). Higher: better recall and robustness, more memory (about 8·M bytes per vector for links) and slower inserts. Allowed 2 to 2048; **8 to 48 is sensible**. Below 8, graphs of many identical vectors can leave some unreachable until upkeep. |
| `ef_construction` | 200 | How widely each insert searches for good neighbors. Higher: a better graph and recall, slower inserts. At least 1; 100 to 400 is typical. |
| `seed` | 42 | Seed of the random level of each node. The same seed and the same operations give the same graph on the same machine (different CPUs' SIMD instructions round distances slightly differently, so graphs may differ across machines). |
| `repair_on_remove` | `true` | When a vector is removed, its neighbors re-choose their links so the graph stays well connected. Keeps search quality high under heavy churn; costs some time per removal. Turn it off only for bulk deletions you follow with `compact()` or `upkeep()`. |
| `quantization` | `QuantizationParams{}` | Compression of the stored vectors: see below. |

Invalid values throw `std::invalid_argument` from the constructor.

### 4.2 `QuantizationParams`

Compression ("quantization") stores each vector as a compact **code** and searches with codes, optionally checking the best candidates again at full precision (**re-ranking**).

| Field | Default | What it does |
|---|---|---|
| `codec` | `CodecKind::None` | `None`: full precision only. `SQ8`: one byte per dimension (4× smaller than floats), very little recall lost. `Binary`: one bit per dimension (32× smaller), coarse: use with re-ranking. `PQ` (product quantization): one byte per *part* of the vector (with the default of one part per 8 dimensions, 32× smaller than floats); needs training, and re-ranking for good recall. `F16` and `BF16` (half precision): two bytes per dimension (2× smaller), no training; with `keep_full_vectors = false` they replace the float32 vectors from the first insert. Recall@10 within 0.005 of float32 in our tests. `F16` (IEEE float16) keeps about 3.3 significant digits and refuses values beyond ±65504 (`std::invalid_argument`, before any change); `BF16` (bfloat16) keeps float32's range with about 2.4 digits. `RaBitQ`: one bit per dimension plus two numbers per vector (8 bytes), with RaBitQ's estimator, whose error shrinks as 1/sqrt(dimensions); see the user guide for when it beats `Binary`. |
| `train_after` | 1000 | Train the codec automatically once this many vectors are stored; until then searches use full precision. 0: only when you call `train_codec()`. |
| `rerank` | `true` | Re-rank the best candidates found with codes at full precision. Better recall for a little more work. Needs the full vectors kept. |
| `rerank_factor` | 4 | Exact searches with re-ranking compare k × this many candidates by code first (at least 1). |
| `truncate_to` | 0 | **Matryoshka truncation:** index and search only the first `truncate_to` dimensions (re-normalized for cosine), for embeddings trained to work when shortened (some OpenAI and Nomic models). Vectors and queries are still given in full, and read back in full. 0, or the full dimension: off. |
| `keep_full_vectors` | `true` | Keep each vector exactly as given. `false` saves memory: reads return the stored form (normalized for cosine, truncated when truncating), and **with a codec the full vectors are dropped once it is trained**, keeping only codes: re-ranking is then off, reads return decoded (approximate) vectors, and the first training is final (training again from decoded vectors could only lose precision). |
| `pq_subspaces` | 0 | PQ only: how many parts each vector is split into (one byte each). 0: one part per 8 dimensions. At most the indexed dimension. |

**Limitations:** binary codes suit only data whose signs carry the meaning (many embedding models do; check recall); PQ codes alone (without re-ranking) recall poorly; SQ8's range is learned when training, so values outside it are clipped until the codec is retrained (see [rebuilds](#47-health-upkeep-compaction-and-rebuilds)).

### 4.3 Creating an index

```cpp
HnswIndex(std::size_t dim, Metric metric, HnswParams params = {}, Schema schema = Schema::dynamic());
```

Creates an empty index for `dim`-dimensional vectors. Throws `std::invalid_argument` for `dim` 0 or above 16,777,216 (2²⁴), or invalid parameters. For crash-safe use, see [`open_durable`](#49-saving-loading-and-durability); to load a saved index, [`load`](#49-saving-loading-and-durability).

### 4.4 Adding, changing and removing vectors

| Method | What it does |
|---|---|
| `void add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata = {})` | Inserts a vector under `id`. Throws `std::invalid_argument` for a wrong dimension, NaN or infinity, an ID already stored, or invalid metadata. |
| `void add_batch(ids, vectors, threads = 0, options = {})` and `add_batch(ids, vectors, metadata, threads = 0, options = {})` | Inserts many vectors at once: `vectors` holds `ids.size()` vectors back to back; `metadata`, if given, has one entry per vector. **Everything is checked first** (dimensions, values, IDs against the index and within the batch, metadata); any problem throws and nothing changes. The graph is then built on `threads` threads (0: all cores). With one thread the result is identical to calling `add()` in order; with more, the graph differs from run to run, but not its quality. Searches wait and then see all of the batch or none of it. `options` adds a deadline, cancellation and progress reports; a cancelled batch is undone and throws `Cancelled`. |
| `bool update(std::uint64_t id, std::span<const float> vector)` | Replaces the vector of `id`, **keeping its metadata**. The new version is linked in a fresh slot while the old one stays searchable, then swapped in: a failure leaves the old vector exactly as it was. Returns `false` if `id` is not stored. Invalid vectors throw even for an unknown ID. |
| `bool update(id, vector, const Metadata& metadata)` | Replaces the vector **and its whole metadata** (fields not given are cleared). |
| `bool upsert(std::uint64_t id, std::span<const float> vector)` | Adds, or replaces the vector keeping its metadata. Returns `true` if added. |
| `bool upsert(id, vector, const Metadata& metadata)` | Adds with this metadata, or replaces vector and whole metadata. |
| `bool remove(std::uint64_t id)` | **Really deletes** the vector: repairs the graph around it (if `repair_on_remove`), frees the ID for reuse, and puts its memory slot on a free list for the next insert. Returns `false` if not stored. If memory runs out during the repair, the vector stays stored and the graph valid. |
| `std::size_t remove_batch(std::span<const std::uint64_t> ids)` | Removes every listed ID that is stored (unknown and repeated IDs skipped). Always completes. Returns how many. |
| `std::size_t remove_where(const Filter& filter)` | Removes every vector whose metadata matches. The filter is evaluated once, before anything changes; searches see the index before or after, never in between. Always completes once the filter is accepted. |
| `bool set_metadata(std::uint64_t id, const Metadata& metadata)` | Changes only the given fields (`unset()` clears one). The vector and graph are untouched. Returns `false` if not stored. |
| `std::size_t set_metadata_where(const Filter& filter, const Metadata& changes)` | Applies `changes` to every match, all or nothing. Needs extra memory about the size of the index's metadata while it runs (the change is made on a copy, then swapped in). |
| `std::size_t count_where(const Filter& filter) const` | How many vectors match: a preview of the two above. |

**Limitations:** removed vectors' slots are reused by later inserts but are not returned to the operating system until `compact()`; under heavy replace-and-remove churn the graph degrades slowly (see [upkeep](#47-health-upkeep-compaction-and-rebuilds), or use a `Collection`, which maintains itself).

### 4.5 Reading records

| Method | What it does |
|---|---|
| `bool contains(std::uint64_t id) const` | Whether `id` is stored. |
| `std::optional<std::vector<float>> get_vector(std::uint64_t id) const` | The vector exactly as given (for cosine, before normalization), unless `keep_full_vectors` is off (then the stored or decoded form). Nothing if not stored. |
| `std::optional<Metadata> get_metadata(std::uint64_t id) const` | The metadata of `id`. |
| `std::optional<Record> get(std::uint64_t id) const` | Vector and metadata together: `Record { id; vector; metadata; }`. |
| `Page scan(const ScanOptions& options = {}) const` | Lists records **a page at a time, in ascending ID order**, optionally only those matching a filter. Pass the returned `Page::next` as `ScanOptions::after` to get the next page; it is empty on the last page. Changes between pages never repeat or skip a record that stays stored. Each page costs one pass over the index. |

```cpp
struct ScanOptions {
    std::optional<Filter> filter;          // only matching records (all if empty)
    std::size_t limit = 100;               // records per page, at least 1 (0 throws)
    std::optional<std::uint64_t> after;    // continue after this ID
    bool with_vectors = true;
    bool with_metadata = true;
};
```

### 4.6 Searching

```cpp
std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t ef = 64) const;
std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t ef,
                                 const SearchOptions& options) const;
```

Returns up to `k` approximately closest vectors, closest first. **`ef`** is the beam width of the search: it is raised to at least `k`; higher values find the true neighbors more reliably and cost more time (64 to 256 is typical; measure with `health()`). `k` larger than the index is capped at its size. A query with the wrong dimension, NaN or infinity throws `std::invalid_argument`.

```cpp
struct SearchOptions {
    std::size_t probes = 0;                         // FlatIndex with IVF: lists to probe (0: the index's default)
    std::optional<Filter> filter;
    std::function<bool(std::uint64_t)> predicate;
    Strategy strategy = Strategy::Auto;
    std::optional<PlannerParams> planner;
    SearchStats* stats = nullptr;
    std::size_t range_expand_outside = 1;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
};
```

| Field | What it does |
|---|---|
| `filter` | Only vectors whose metadata matches are returned. **You get k results if at least k match** (the search widens as needed). |
| `predicate` | Any condition on the ID, for data outside the index (permissions, say). With a filter, both must pass (the filter is checked first). Must be safe to call from several threads (batch searches call it concurrently), and must not change any index. |
| `strategy` | `Auto` (default): the **query planner** chooses. `ForceExact`: scan every matching vector (exact answers). `ForceGraph`: the graph search that scores every vector it visits. `ForceAcorn`: the ACORN graph search (below; without a filter it is the ordinary graph search). |
| `planner` | Planner settings for this query only (otherwise the index's: see `set_planner_params`). |
| `stats` | If set, filled with what the search did (`SearchStats`, below). |
| `range_expand_outside` | Range searches only: how many hops the search may take through vectors *outside* the radius to reach more of the region (more finds more, at more cost). |
| `deadline`, `cancel` | Stop when the moment passes, or when another thread sets the flag: the search returns the best results found so far (sorted and valid, possibly incomplete) and sets `stats->stopped_early`. Checked every few dozen steps. |
| `modifiers` | Opt-in re-scoring by metadata (below). |
| `diversity` | Opt-in MMR over a candidate pool of max(4k, 20) (or the modifiers' `pool`), dense indexes only (`SparseIndex` throws `std::invalid_argument`): as for [collections](#27-searching). Results then carry `score` (their relevance, or the modified score). |

**How filtered searches are planned.** A filter that matches few vectors would make a graph search walk through many non-matching ones. So the planner estimates how many vectors match (exactly, from a [payload index](#46-searching), or from a sample of `sample_size` vectors) and:

- **scans exactly** when at most `max_exact_matches` vectors (2000) **or** at most `max_exact_fraction` of them (1%) are expected to match;
- otherwise, below `acorn_below` (20%) matching, searches the graph **the ACORN way**: only matching vectors are scored and enter the beam, and a neighbor that fails the filter is looked through to its own neighbors (two hops) without computing its distance, which keeps the graph connected as the filter sees it. Its beam is `ef` × `acorn_ef_multiplier` (4). If it finds fewer than k matches (they lie beyond two hops of where it went), it falls back to the next strategy, so results are never worse than without it;
- otherwise **searches the graph**, scoring every vector it visits, with `ef` multiplied by 1/selectivity, up to `max_ef_multiplier` (32) times.
- **Exact fallback:** a graph or ACORN search the planner chose stops once it has computed more distances than `fallback_factor` (1) times the estimated matches (at least 4 × `ef`), and scans the matches exactly instead (`PlanReason::ExactFallback`, `strategy` `ForceExact`). This bounds filters correlated with the vectors, whose matches sit in a few regions the graph search reaches only after crossing much of the graph: measured on 30,000 vectors with a filter matching 10% of them by cluster, 16,876 distances and 5.9 ms became 3,803 and 1.4 ms (4.3× faster; 6.2× on an indexed field), recall unchanged. Results are the exact search's. Forced strategies are never changed; `fallback_factor = 0` turns it off.

Measured on 20,000 clustered 32-dimensional vectors, recall@10 against exact search, ACORN against the widened graph search:

| Matching | Matches spread evenly | Matches clustered (ACORN's hard case) |
|---|---|---|
| 5% | 42× fewer distance computations, recall 1.000 | 1.7× fewer, recall 0.980 (graph: 1.000) |
| 10% | 17× fewer, recall 1.000 | 1.6× fewer, recall 0.997 |
| 20% | 3.8× fewer, recall 1.000 | 1.4× fewer, recall 0.997 |

ACORN applies by default only where the exact scan would cost more than 2,000 distances and between 1% and 20% match, so on indexes of about 10,000 vectors or more.

```cpp
struct PlannerParams {
    std::size_t max_exact_matches = 2000;
    double max_exact_fraction = 0.01;
    std::size_t sample_size = 256;
    double max_ef_multiplier = 32.0;
    double acorn_below = 0.2;          // 0: never ACORN (the graph search, as before)
    double acorn_ef_multiplier = 4.0;  // at least 1
    double fallback_factor = 1.0;      // graph or ACORN past this many times the estimated matches: exact (0: never)
};
```

On very large indexes, lower `max_exact_fraction`: 1% of 100 million vectors is a million exact distance computations per query. Invalid values throw `std::invalid_argument` (`validate(const PlannerParams&)` checks them without an index).

Advanced: `plan_search(capacity, live, eligible, exact_matches, ef, forced, params)` is the planner itself, exposed for tools and tests: given the index's slot count, live count, an eligibility test, an exact match count if known, the beam width, a forced strategy and the settings, it returns a `PlanDecision` with `strategy` (what to run), `reason`, `selectivity` (estimated matching fraction, 0 to 1), `estimated_matches`, `exact_count` (whether the count is exact) and `ef` (the beam width to use). Applications normally read the same figures from `SearchStats` instead.

```cpp
struct SearchStats {
    Strategy strategy;              // ForceExact, ForceAcorn or ForceGraph: what actually ran (ForceGraph after an ACORN fallback)
    PlanReason reason;              // NoFilter, Forced, MatchCount, MatchFraction, Acorn, Graph, FlatIndex or ExactFallback
    double selectivity;             // estimated fraction of vectors matching
    std::size_t estimated_matches;
    bool exact_count;               // the estimate came from a payload index
    std::size_t ef;                 // beam width a graph search used
    std::size_t nodes_visited, distance_computations;
    bool stopped_early;             // a deadline or cancel flag stopped it: results may be incomplete
};
```

Other searches:

| Method | What it does |
|---|---|
| `search_batch(queries, k, ef = 64, threads = 0, options = {})` | Many queries at once (`dim` floats each, back to back), on `threads` threads (0: one per hardware thread). Results are identical to separate `search()` calls. Every query is checked before any work starts. `options.stats` is ignored. |
| `search_range(query, radius, max_results = kNoLimit, options = {})` | Every vector within `radius` (in distance units: **squared** for L2, use `l2_radius(r)`; negative for inner product), closest first, at most `max_results`. Approximate: grows a region from the nearest vectors through graph links (see `range_expand_outside`). `Strategy::ForceExact` scans everything instead. A NaN radius throws. |
| `search_by_id(id, k, ef = 64, options = {}, include_self = false)` | "More like this": searches with the vector stored for `id`, which cannot change in between. The record itself is left out unless `include_self`. Nothing if `id` is not stored. |
| `void create_payload_index(std::string_view field)` | Keeps exact value counts and **posting lists** for a **keyword, boolean or tag-set** field: the planner counts matches exactly, and filters on it read only the matching rows (much faster when few match), as do facets. Worth it for fields you filter or facet on often. Other field types throw `std::invalid_argument`. Collections have it too. |
| `std::vector<FacetResult> facets(const std::vector<FacetRequest>& requests, const std::optional<Filter>& filter = {}) const` | **Faceted counts**: per request, how many vectors hold each value of a field, over every vector or those matching `filter` (see *Faceted search* below). |
| `std::vector<FacetResult> facets_for(std::span<const std::uint64_t> ids, const std::vector<FacetRequest>& requests) const` | The same over the vectors with these IDs, such as a search's top hits (unknown IDs skipped, repeats counted once). |
| `Schema schema() const`, `std::vector<std::string> payload_indexes() const` | The metadata schema in force (its fields so far, for a dynamic schema), and the fields with a payload index. |
| `void set_planner_params(const PlannerParams& p)`, `PlannerParams planner_params() const` | The index's default planner settings. |

**Score modifiers (opt-in): re-ranking by metadata.** `SearchOptions::modifiers` (also in `HybridOptions` and `CollectionSearchOptions`) re-scores the results by metadata, in C++, with no callback per node:

```cpp
ScoreModifiers m;
m.decays.push_back({"published", double(now_ms()), 30.0 * 86400000});   // halves every 30 days
m.multipliers.push_back({"provenance_confidence"});                     // times a confidence in [0, 1]
m.boosts.push_back({Filter::eq("edge_type", "verified_decision"), 0.5}); // + 0.5 on a match, nothing filtered out
SearchOptions o;
o.modifiers = m;
```

**final score = (base × every decay × every multiplier) + every matching bonus.** The base is the fused score (collections, hybrid search), the dot product (sparse indexes), or a relevance from the distance (dense indexes: 1/(1+d) for L2, 1 − d/2 for cosine, the logistic of the dot product for inner product). Results then carry `SearchResult::score` (collections and hybrid search: `score` and `base_score`) and are ordered by it.

| Rule | Fields | Effect |
|---|---|---|
| `RecencyDecay` | `field` (a time: dates, or any unit), `origin` ("now", same unit), `half_life` (> 0, same unit), `shape` (`Exponential`: 2^(−Δt/half_life); `Linear`: 1 − Δt/(2·half_life), zero at twice the half-life), `missing` (factor without the field, 1), `floor` (smallest factor, 0) | Older records score lower; Δt = origin − value, and values at or after `origin` are not decayed |
| `FieldMultiplier` | `field`, `missing` (1), `min` (0), `max` (∞) | The score times the field's value (booleans 0 or 1), clamped |
| `MatchBoost` | `when` (any `Filter`), `bonus` (may be negative) | Adds the bonus when the record matches: boosting without filtering |
| `ScoreModifiers::pool` | 0: max(4k, 20) | Candidates re-scored before the best k are kept: a boost can promote a result that ranked below k only if it is in the pool. `pool_for(k)` returns the pool a search of k uses. |

Invalid rules (an empty field, a half-life that is not positive, a negative minimum, a floor outside [0, 1], min above max, non-finite numbers) throw `std::invalid_argument` when the search runs. Keyword and tag values count as missing for decays and multipliers.

**Faceted search** (`facets.h`): counts by value of metadata fields ("120 in English, 40 in French"), the same on `HnswIndex`, `FlatIndex` and `Collection`:

```cpp
struct FacetRequest {
    std::string field;
    std::size_t limit = 10;            // the most frequent values (0: all); the rest summed in `other`
    std::vector<double> boundaries;    // integer and float fields: n boundaries make n + 1 range buckets
    bool distinct = false;             // also count the distinct values (exact)
};
struct FacetResult {
    std::string field;
    std::size_t total, missing;        // documents counted, and of those without the field
    std::vector<FacetValue> values;    // {Value value; std::size_t count}: most frequent first, ties by value
    std::size_t other;                 // summed counts of the values beyond `limit`
    std::vector<FacetBucket> buckets;  // {optional<double> from, to; count}: [from, to), open at the ends
    std::optional<std::size_t> distinct;
};
```

Every type of field has values (a document counts once per distinct tag); boundaries need an integer or float field and replace the value list. Every request is checked before anything is counted: an empty field name, boundaries not finite and strictly ascending, boundaries on another type, or an unknown field under a strict schema throw `std::invalid_argument`; under a dynamic schema an unknown field is missing everywhere. A collection never counts expired documents. With a payload index on the field and no filter, counts come from the index without reading documents.

**Limitations:** HNSW is approximate: a true neighbor can be missed (raise `ef`, or `M` at build time). ACORN can miss a few matches when they cluster away from the query (recall 0.98 at 5% matching above): raise `acorn_ef_multiplier`, or set `acorn_below = 0`. Range search is approximate as well. Filters on a field without a payload index are estimated by sampling, so a rare value can be misjudged; the planner then still returns correct results, only more slowly.

### 4.7 Health, upkeep, compaction and rebuilds

| Method | What it does |
|---|---|
| `HealthReport health(const HealthOptions& options = {}) const` | A report on storage, graph structure, an estimate of recall, compression drift and churn. Searches continue meanwhile. See [Health](#91-health-reports). |
| `UpkeepStats upkeep(const UpkeepOptions& upkeep_options = {}, const OperationOptions& options = {})` | Repairs the graph incrementally, like a database's vacuum: replaces links to removed vectors and reconnects vectors no search can reach. Use `max_nodes` or a deadline to keep each call short; searches run in between calls. A deadline stops it only **after at least one repair** in the call, so repeated calls always finish; a cancellation stops it at once. When complete, churn counters reset. See [Upkeep](#92-upkeep). |
| `CompactStats compact(const OperationOptions& options = {})` | Rebuilds storage from the live vectors only: no free slots, memory returned. User IDs are kept. The new index is built completely before replacing the old one, so running out of memory leaves the old one intact. **Blocks searches and changes** while it runs. Returns `{ kept, reclaimed }`. |
| `CompactStats reorder(const OperationOptions& options = {})` | **Opt-in speed-up:** renumbers the nodes so neighbors are stored near each other (grouped by nearest k-means centroid, about 1,024 per group, breadth-first within a group). The graph, results and every record are unchanged; removed slots are dropped, as by `compact()`. Measured on 200,000 clustered 128-dimensional vectors: **about 24% faster searches** (95.5–99.0 → 70.5–76.4 µs per query); on uniformly random data, about 4%. Takes a few seconds per 200,000 vectors (k-means assignment, on all cores). Blocks searches and changes while it runs, so call it after building or bulk loading. All or nothing (cancellation, deadlines, out of memory); logged for durable indexes. |
| `void set_auto_compact(double fraction)`, `double auto_compact() const` | Compact automatically once free slots exceed `fraction` of all slots (0: off, the default; at most 0.9). The change that crosses the threshold then takes as long as a `compact()`. A runtime setting, not saved. |
| `RebuildStats rebuild(const RebuildOptions& rebuild_options = {}, const OperationOptions& options = {})` | Like `compact()`, but **without blocking**: a snapshot is taken, a new index (with the codec retrained if `retrain`) is built while searches and changes continue, then changes made meanwhile are copied in and the new index swapped in. Needs memory for a second copy. One rebuild at a time (`std::logic_error` otherwise). A deadline, cancellation or running out of memory throws `Cancelled` (or `std::bad_alloc`) and leaves the index as it was. |

### 4.8 Compression

| Method | What it does |
|---|---|
| `void train_codec()` | Trains the codec on the stored vectors now and encodes them all; later vectors are encoded as they arrive. Training again replaces the old training. Throws `std::logic_error` without a codec, or once only codes are kept (the first training is final); `std::invalid_argument` if the index is empty. Training also happens automatically (`train_after`). |
| `bool codec_trained() const` | Whether searches use codes. |
| `bool has_vector_values() const` | Whether full-precision values are stored (false once a codes-only index is trained). |
| `std::optional<std::vector<std::uint8_t>> get_code(std::uint64_t id) const` | The code of `id`, for inspection. |
| `const VectorCodec* codec() const` | The codec, for inspection while no other thread changes the index. |

### 4.9 Saving, loading and durability

| Method | What it does |
|---|---|
| `void save(const std::string& path, const OperationOptions& options = {}) const` | Saves everything (vectors, graph, metadata, settings) to one file. Written to a temporary file and renamed into place, so a failure never damages an existing file; concurrent saves to one path each produce a complete file (the last wins). Searches continue meanwhile; changes from other threads wait. Throws `std::runtime_error` if the file cannot be written, `Cancelled` if a deadline or cancellation stops it (no file is left behind). |
| `void save(std::ostream& out, const OperationOptions& options = {}) const` | The same, to a stream. |
| `static std::unique_ptr<HnswIndex> load(const std::string& path)`, `load(std::istream& in)` | Loads a saved index. Throws `FormatError` for a damaged or truncated file, an unsupported version, or a file of another kind; `std::runtime_error` if it cannot be opened. A failed load never produces a partly built index. Files from older versions of the library load; saving writes the current version. A stream must be seekable (files and string streams are). |
| `static std::unique_ptr<HnswIndex> open_durable(dir, dim, metric, params = {}, schema = Schema::dynamic(), options = {}, stats = nullptr)` | Opens a **crash-safe** index in a directory: created, or recovered from its snapshot and log (see [Durability](#8-durability-write-ahead-logs-and-snapshots)). When recovering, `dim` and `metric` must match (`std::invalid_argument`); the stored settings apply. One process per directory (`std::runtime_error`). |
| `std::uint64_t checkpoint()` | Durable indexes: a new log segment and a snapshot (searches continue; changes wait while it is written). Also heals a failed log. Throws `std::logic_error` for a non-durable index. **Call it from time to time** (a `Collection` with maintenance does it for you), or the log grows without limit. |
| `bool durable() const` | Whether the index was opened with `open_durable`. |
| `attach_log(WalWriter)`, `detach_log()`, `replay(records)` | Advanced: log a non-durable index's changes yourself. See [8.5](#85-low-level-log-access). |

**Limitations:** files are portable between operating systems and CPUs (little-endian, checksummed). A plain `save()` does not force data onto the physical disk; durable indexes do. Saving takes a read lock for its whole duration, so changes wait for it.

### 4.10 Inspecting the index

| Method | Returns |
|---|---|
| `size()` | live vectors |
| `deleted_count()` | free slots of removed vectors, waiting for reuse |
| `capacity()` | all slots, live or free |
| `dim()` | the dimension vectors and queries are given in |
| `indexed_dim()` | the dimensions actually stored and searched (fewer with Matryoshka truncation) |
| `metric()`, `params()` | settings |
| `max_level()`, `entry_point()` | the graph's top level and where searches start (−1 and `kEmpty` when empty) |
| `metadata()`, `storage()` | read-only internals, for tests and tools; only while no other thread changes the index |
| `pooled_visited_lists()` | search scratch buffers kept for reuse |

---

## 5. FlatIndex: exact search, optionally with an inverted file

`#include "flat_index.h"`

Stores vectors in one packed array and answers searches by comparing the query with **every** vector: always exact, no build cost, the fastest inserts, and good speed up to roughly 100,000 vectors (SIMD makes each comparison cheap). With an **inverted file (IVF)**, vectors are grouped into lists around learned centroids and a search scans only the closest lists: much faster on large data, no longer exact.

```cpp
FlatIndex(std::size_t dim, Metric metric, Schema schema = Schema::dynamic(),
          QuantizationParams quantization = {}, IvfParams ivf = {});
```

Note the **argument order differs from `HnswIndex`**: the schema comes third, then compression, then IVF.

### 5.1 `IvfParams`

| Field | Default | What it does |
|---|---|---|
| `lists` | 0 | Number of lists (centroids). 0: no IVF (every search scans everything). A common choice is about the square root of the number of vectors. At most 2³¹. |
| `probes` | 8 | Lists scanned per query. More: better recall, slower. `SearchOptions::probes` overrides it per query. |
| `train_after` | 10000 | Train the lists automatically once this many vectors (and at least `lists`) are stored; until then every search scans everything. 0: only when you call `train_lists()`. |

### 5.2 Methods

`FlatIndex` has **the same methods as `HnswIndex`** for adding, changing, removing, reading, scanning, bulk changes, payload indexes, compression, health, rebuilds, saving, loading and durability (see [section 4](#4-hnswindex-the-graph-index)), with these differences:

| Method | Difference |
|---|---|
| `search(query, k)`, `search(query, k, options)` | No `ef`: the search is exact (or scans `probes` lists with IVF). With a filter, only matching vectors are compared. |
| `search_batch(queries, k, threads = 0, options = {})` | No `ef`. Large batches compare queries against blocks of vectors at once, which is faster than separate searches; results are identical. |
| `search_range(...)`, `search_by_id(id, k, options = {}, include_self = false)` | Exact (without IVF). No `ef`. |
| `remove(id)` | Moves the last vector into the removed one's place ("swap with last"), so storage never has holes. |
| `compact()` | Nothing to reclaim in the vectors (no holes); only cleans up unused metadata strings. Reports 0 reclaimed. |
| No `upkeep()`, `set_auto_compact()` or planner settings | There is no graph to maintain or plan around. |
| `void train_lists()` | Trains the IVF centroids (k-means) on the stored vectors now and assigns every vector to its nearest list; later vectors are assigned as they arrive. Training again replaces the lists. Throws `std::logic_error` without `IvfParams::lists`, `std::invalid_argument` with fewer vectors than lists. |
| `bool lists_trained() const` | Whether searches use the lists. |
| `std::optional<std::uint32_t> list_of(std::uint64_t id) const` | The list holding `id` (nothing if untrained or not stored). |
| `std::vector<std::size_t> list_sizes() const` | Vectors per list (empty if untrained). Very uneven sizes make searches slower; `health()` reports the imbalance, and `rebuild()` retrains the lists. |
| `const IvfParams& ivf() const`, `const QuantizationParams& quantization() const` | Settings. |

**Limitations:** without IVF, search time grows linearly with the number of vectors. With IVF, recall depends on `probes`, and lists trained on old data fit new data worse over time (`health()` shows it as centroid drift; `rebuild()` retrains). IVF with PQ codes and no re-ranking recalls poorly.

---

## 6. DiskIndex: indexes larger than memory

`#include "disk_index.h"`

A DiskANN-style index: the graph and the full vectors stay **on disk**, in 4 KB blocks; memory holds only compact PQ codes (which guide the search), IDs and metadata. A search reads a few dozen blocks from disk. Every block is checksummed and verified when read.

**It is read-only:** built once, then opened and searched. To change it, build it again.

### 6.1 Building

```cpp
static void DiskIndex::build(const std::string& path, std::span<const std::uint64_t> ids, std::span<const float> vectors,
                             std::size_t dim, Metric metric, const DiskIndexParams& params = {},
                             std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic());
```

Builds an index over `vectors` (`ids.size() × dim` values) and writes it to `path`, replacing any file there only once complete. **The vectors must fit in memory** while building. `metadata`, if given, has one entry per vector. Throws `std::invalid_argument` for invalid input (sizes, repeated IDs, non-finite values, settings).

```cpp
static void DiskIndex::build_from_file(const std::string& path, const std::string& vectors_path, Metric metric,
                                       const LargeBuildParams& params = {}, std::span<const std::uint64_t> ids = {},
                                       std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic(),
                                       LargeBuildStats* stats = nullptr);
void write_vectors_file(const std::string& path, std::span<const float> vectors, std::size_t dim);
```

Builds from a **vectors file** that need not fit in memory: the `.fbin` layout used by DiskANN and the big-ANN benchmarks (a 32-bit count, a 32-bit dimension, then the floats, little-endian), which `write_vectors_file` writes. If the data fits the memory budget it is built in memory; otherwise it is divided into partitions that fit, each vector joining its `replicas` nearest partitions, a graph is built per partition, and the graphs are merged. Temporary files go to a directory next to `path` and are removed afterwards. IDs default to 0, 1, 2, … in file order.

| `DiskIndexParams` field | Default | What it does |
|---|---|---|
| `graph.R` | 64 | Most neighbors per node. More: better recall, more disk space and reads. |
| `graph.L` | 100 | Beam width while building (at least `R`). |
| `graph.alpha` | 1.2 | Pruning strength (at least 1): higher keeps more long-range links. |
| `graph.seed` | 1 | Same seed, same graph. |
| `pq_subspaces` | 0 | Parts of the in-memory codes; 0: one per 2 dimensions. Finer codes than PQ's usual default, because here they decide which blocks to read (measured: 32 dimensions with 4 parts reached recall 0.71, with 16 parts 1.00, for the same reads). |
| `vectors` | `DiskVectorFormat::Float32` | How node vectors are stored: `Float32` (exact), `Float16` (values up to 65504 in magnitude; L2 builds beyond it are refused) or `BFloat16` (the float32 range). Half precision makes files about 29% smaller and, where a float32 record spans two 4 KB blocks (from about 1000 dimensions with R = 32), halves the blocks a search reads; measured recall unchanged. Half-precision files are format version 2. `DiskIndex::vector_format()` reports it. |

| `LargeBuildParams` field | Default | What it does |
|---|---|---|
| `index` | `DiskIndexParams{}` | As above. |
| `memory_budget` | 1 GiB | Memory for vectors and graphs while building one partition. The codes, IDs and metadata come on top. |
| `replicas` | 2 | Partitions each vector joins (at least 2 when there are several), so that neighboring partitions' graphs connect. |
| `partitions` | 0 | 0: automatic. Explicit values too small for the budget are rejected. |

`LargeBuildStats` reports `partitions` built (1: built in memory), `largest_partition`, and `merged_and_pruned` vectors.

### 6.2 Opening and searching

```cpp
static std::unique_ptr<DiskIndex> DiskIndex::open(const std::string& path, const DiskOpenOptions& options = {});
std::vector<SearchResult> search(std::span<const float> query, std::size_t k, const DiskSearchParams& params = {},
                                 const SearchOptions& options = {}, DiskSearchStats* stats = nullptr) const;
```

`open` verifies everything it loads into memory and throws `FormatError` for a damaged or invalid file. The file may be deleted or replaced while open; reads continue from the old one.

| `DiskOpenOptions` field | Default | What it does |
|---|---|---|
| `io_threads` | 0 | Threads that read blocks concurrently, so a search's reads overlap instead of following one another (helps most on SSDs and network storage). 0: reads on the searching thread. |
| `simulated_latency` | 0 | Delays every read, like slow storage: for tests and benchmarks. |

| `DiskSearchParams` field | Default | What it does |
|---|---|---|
| `L` | 100 | Candidate list size (at least k is used): higher recall, more reads. |
| `beam_width` | 4 | Blocks read per round (W): fewer round trips against more reads. |

A search ranks candidates by PQ distance in memory, reads the best unread ones from disk `beam_width` per round, computes their exact distances, and follows their neighbors, until the best `L` candidates have all been read. **Distances in results are exact.** Filters and predicates work: you get k results if k match (the search widens as needed), and a filter matching at most max(L, k) vectors skips the graph and reads exactly those (an exact answer). Deadlines and cancellation are checked every round. Throws `FormatError` if a block read is damaged.

`DiskSearchStats` reports `nodes_read`, `blocks_read`, `rounds` and `stopped_early`.

`BlockFile` is the positional file reader underneath (`read(offset, buffer, n)`, `size()`), safe from many threads at once; `set_simulated_latency(std::chrono::microseconds)` delays every read, for tests and benchmarks (the same as `DiskOpenOptions::simulated_latency`).

Other methods: `search_batch(queries, k, params, threads = 0, options)`; `search_by_id(id, k, params, options, include_self = false)`; `contains`, `get_vector`, `get`, `scan` (records read from disk on demand); `health(options)` (reads every node once; churn is always 0); `size`, `dim`, `metric`, `max_degree`, `medoid`, `id_of(node)`, `node_of(id)`, `codec()`, `metadata()`, `read_node(node)` (one node as stored, for tools).

**Limitations:**
- **Read-only.** No adds, updates or removals: build again.
- **L2 and cosine only** (no inner product).
- Vectors read back are as stored: **normalized for cosine** (the disk index keeps no original copy).
- `build()` needs the vectors in memory; use `build_from_file` for larger data.
- Search speed depends on the storage: each search reads tens of 4 KB blocks.

---

## 7. Sparse vectors, BM25 and hybrid search

`#include "sparse.h"`

For keyword search on its own, or to combine your own dense and sparse indexes by hand. (A [`Collection`](#2-collection-the-recommended-starting-point) does all of this for you.)

### 7.1 `SparseVector`

```cpp
struct SparseVector {
    std::vector<std::uint32_t> indices;   // terms
    std::vector<float> values;            // their weights
};
double sparse_dot(const SparseVector& a, const SparseVector& b);   // dot product over shared terms
```

A sparse vector lists only its nonzero entries: **term** numbers and their **weights**. Order does not matter; terms must be distinct; weights finite; zero weights are dropped when stored. Similarity is the **dot product** over shared terms.

### 7.2 `SparseIndex`

An **inverted index**: for each term, the list of documents containing it. Search is **exact**: it adds up query weight × document weight over the query's terms. Its cost is the total length of those terms' lists, not the size of the index.

`SparseIndex(Schema schema = Schema::dynamic())` creates an empty index. It has the record methods of the dense indexes, with sparse vectors:

| Method | What it does |
|---|---|
| `add(id, vector, metadata = {})` | Adds a document. Throws `std::invalid_argument` for a duplicate ID, mismatched lengths, repeated terms, non-finite weights, or invalid metadata. |
| `remove(id)`, `remove_batch(ids)`, `remove_where(filter)` | Removal. |
| `update(id, vector)`, `update(id, vector, metadata)` | Replace the vector (keeping metadata), or both. `false` if unknown. |
| `upsert(id, vector)`, `upsert(id, vector, metadata)` | Add or replace; `true` if added. |
| `set_metadata(id, metadata)` | Change only the given fields. |
| `contains`, `size`, `get_vector`, `get_metadata` | Reading. `get_vector` returns the terms sorted. |
| `std::vector<SparseRecord> records() const` | Every record (`id`, `vector`, `metadata`) in ID order. |
| `static std::unique_ptr<SparseIndex> open_durable(dir, schema = dynamic, options = {}, stats = nullptr)` | A **durable** sparse index in `dir`, as for `HnswIndex::open_durable`: created if new, or recovered (the newest snapshot, then the log after it). Every change is logged before it is applied (`remove_where` as the IDs it removed); a second process is refused while this one holds `dir`. |
| `std::uint64_t checkpoint()` | Durable indexes: writes a snapshot and drops the log it covers. Throws `std::logic_error` otherwise. |
| `bool replay(std::span<const WalRecord> records)` | Re-applies logged changes (recovery uses it); true if an unfinished last one was dropped. |
| `Schema schema() const`, `std::vector<std::string> payload_indexes() const` | The metadata schema in force, and the indexed fields (none). |
| `search(query, k, options = {})` | The `k` documents with the highest dot product, best first. `distance` is **minus** the score (smaller is better, as everywhere). Only documents sharing at least one term with the query are candidates. Honors `filter`, `predicate`, `deadline`, `cancel` and `stats` from `SearchOptions`; other options do not apply. |
| `document_frequency(term)` | Documents containing `term` (used for BM25's IDF). |
| `terms()`, `postings()`, `dead_postings()`, `slots()` | Distinct terms; posting entries (live and awaiting compaction); entries of removed documents; document slots. |
| `compact()` | Drops removed documents from the lists. Runs automatically once removed entries or slots outnumber live ones. |
| `save(path)`, `save(out)`, `load(path)`, `load(in)` | One file, checksummed. Saving writes the compacted form, so the bytes do not depend on pending removals. |

**Limitations:** search is exact without pruning, so very common terms (stop words) make long lists and slow searches: drop them for large text collections. A standalone `SparseIndex` has **no write-ahead log** (a `Collection` logs its sparse sides). Term numbers are 32-bit.

### 7.3 Searching combined dense and sparse indexes by hand

```cpp
template <class DenseIndex>
std::vector<HybridResult> hybrid_search(const DenseIndex& dense, std::span<const float> dense_query,
                                        const SparseIndex& sparse, const SparseVector& sparse_query,
                                        std::size_t k, const HybridOptions& options = {});
std::vector<HybridResult> fuse(std::span<const SearchResult> dense, std::span<const SearchResult> sparse,
                               std::size_t k, const HybridOptions& options = {});
```

`hybrid_search` searches a dense index (`HnswIndex` or `FlatIndex`) and a `SparseIndex` that hold **the same IDs**, then fuses the two rankings. `fuse` fuses two result lists you already have.

| `HybridOptions` field | Default | What it does |
|---|---|---|
| `fusion` | `Fusion::Rrf` | Reciprocal Rank Fusion, or `Fusion::Weighted` (scores scaled to [0, 1] per list): as described for [collections](#27-searching). |
| `rrf_k` | 60 | The RRF constant. |
| `dense_weight`, `sparse_weight` | 1 | Weight of each list (finite, non-negative). |
| `dense_k`, `sparse_k` | 0 | Candidates per side; 0: max(4·k, 20). |
| `dense_ef` | 0 | HNSW beam width; 0: max(dense candidates, 64). |
| `filter` | none | Applied to both sides, in addition to any filter in the per-side options. Each index evaluates it against **its own** metadata, so both need it. |
| `dense_options`, `sparse_options` | `{}` | Per-side `SearchOptions`. |

`HybridResult` has `id`, `score` (fused, higher is better), `dense_rank` and `sparse_rank` (1-based, if found), `dense_distance` and `sparse_score`. Negative or non-finite weights or `rrf_k` throw `std::invalid_argument`; `k = 0` returns nothing.

### 7.4 BM25

```cpp
struct Bm25Params {
    double k1 = 1.2, b = 0.75, avg_doc_length = 100.0;
    bool unicode = false, stop_words = false, stem = false;  // text analysis, off by default
};
class Bm25 { /* see below */ };
```

`Bm25` turns text into sparse vectors whose dot product **is the BM25 score**, the classic keyword-relevance formula used by search engines.

| Parameter | What it does |
|---|---|
| `k1` (1.2) | How quickly repeated words stop adding to a document's score (at least 0). |
| `b` (0.75) | How much long documents are penalized (0 to 1; 0: not at all). |
| `avg_doc_length` (100) | Average document length in tokens, for that penalty (greater than 0). Set it with `fit()` or from your data. |
| `unicode` (false) | Tokens by `tokenize_unicode()` instead of `tokenize()`: UTF-8 decoded, lowercased beyond ASCII (Latin-1, Latin Extended-A, Greek, Cyrillic), Unicode punctuation, spaces, symbols and emoji separating words, invalid UTF-8 never inside a token, and an apostrophe between letters kept (`don't`). |
| `stop_words` (false) | Leaves out Snowball's English stop words (174: `the`, `and`, `don't`, ...). |
| `stem` (false) | Reduces tokens of letters a to z (and apostrophes) to their Snowball English stems: `running`, `runs` and `run` match. |

| Method | What it does |
|---|---|
| `Bm25(Bm25Params params = {})` | Throws `std::invalid_argument` for values outside the ranges above. |
| `static std::vector<std::string> tokenize(std::string_view text)` | Lowercase runs of letters and digits; everything else separates tokens. Bytes of UTF-8 characters count as letters, so accented and non-Latin words stay whole (but are not lowercased). |
| `std::vector<std::string> analyze(std::string_view text) const` | The terms of `text` as the parameters ask: `tokenize()` or `tokenize_unicode()`, then stop words left out, then stems. `fit` and both encoders use it. |
| `static std::uint32_t term_id(std::string_view token)` | The token's term number: a 32-bit hash (FNV-1a). |
| `void fit(std::span<const std::string_view> documents)` | Sets `avg_doc_length` to the mean token count of `documents` (unchanged if they have no tokens). |
| `SparseVector encode_document(std::string_view text) const` | The document's term weights (the term-frequency part of BM25). |
| `SparseVector encode_query(std::string_view text, const SparseIndex& index) const` | The query's term weights: each term's IDF, `ln(1 + (N − df + 0.5) / (df + 0.5))`, from the index's current document counts. Terms no document has are dropped. |

**Text analysis** (`text_analysis.h`): `stem_english(word)` is the Snowball English stemmer (Porter2, from Snowball's `english.sbl`), equal to Snowball's own output on all 42,649 words of its test vocabulary; `tokenize_unicode(text)` and `is_english_stop_word(token)` are as above. In a collection the analysis is fixed at creation: saved with it (an optional block, so collections without analysis are written exactly as before, and older versions refuse the block cleanly), and reopening a durable collection with other analysis throws `std::invalid_argument`, since its documents' terms would not match the queries'.

**Limitations:** no synonyms, and stemming and stop words for English only; two different words with the same 32-bit hash merge (rare, and it only blends their counts). Document weights use `avg_doc_length` as it was when they were encoded: changing it later does not update stored documents. Query weights use the index's counts at query time, so encode queries against the index you search.

---

## 8. Durability: write-ahead logs and snapshots

`#include "durable.h"` (included by the index and collection headers)

A **durable** index or collection (`HnswIndex::open_durable`, `FlatIndex::open_durable`, `Collection::open`) survives crashes: every change is appended to a **write-ahead log** before it is applied, and **snapshots** (checkpoints) capture the whole state from time to time so the log can be trimmed. After a crash, opening the directory again loads the newest snapshot and replays the log since. The result is the same records as before the crash, and, where building is deterministic, the same index byte for byte.

### 8.1 `DurableOptions` and `Durability`

| `DurableOptions` field | Default | What it does |
|---|---|---|
| `keep_snapshots` | 2 | Snapshots kept (at least 1). Older ones are fallbacks if the newest cannot be read; log segments are kept as long as a kept snapshot needs them. |
| `durability` | `Durability::Grouped` | When logged changes reach the disk itself (table below). |
| `group_interval` | 10 ms | `Grouped` mode: how often the background flush runs (at least 1 ms). |

Every mode writes each change to the **operating system** immediately, so a crash of your **process** never loses a change that returned. The modes differ in when data reaches the **physical disk**, which is what survives a **power loss** or an operating-system crash:

| `Durability` | Data reaches the disk | Survives | Cost (measured on a virtual disk; real disks differ) |
|---|---|---|---|
| `Buffered` | at checkpoints and when closed | a process crash | about 3% slower than not durable |
| `Grouped` (default) | also every `group_interval`, from a background thread | a power loss, except changes of the last interval | about 13% slower |
| `Synced` | before each change is applied | a power loss, entirely | 2.4× slower here; on real disks a flush costs 0.1 to several milliseconds, so it can be 10 to 50× slower. Run `bench_durability` on your storage. |

### 8.2 `RecoveryStats`

Filled by `open_durable` and `Collection::open` when you pass a pointer:

| Field | Meaning |
|---|---|
| `created` | A new directory was created (nothing recovered). |
| `snapshot_lsn` | The sequence number of the snapshot recovery started from. |
| `segments` | Log segments read. |
| `replayed` | Log records replayed (aborted ones included). |
| `fallbacks` | Newer snapshots that could not be read (an older one was used). |
| `segments`, `replayed` | Log segments read; records replayed. |
| `torn_bytes` | Bytes of a last record cut short by the crash, discarded. |
| `dropped_unfinished` | The last logged operation was in progress when the process stopped (logged, failed, and stopped before its failure was recorded); it was dropped, and a checkpoint taken. |

### 8.3 What recovery accepts and refuses

- A record cut short **at the very end** of the log (a crash in the middle of a write) is discarded: normal after a crash.
- If the newest snapshot cannot be read, the previous one is used, with the log since it.
- **Refused** with `FormatError`, because a crash cannot cause them: damage in the middle of the log, a missing log segment, a segment from another index, a damaged directory identity file. Such damage means the storage itself is failing or files were changed by hand.

### 8.4 The directory

| File | Contents |
|---|---|
| `DURABLE` | The index kind and a random identity, written once. Every log segment carries the identity, so segments can never be mixed between indexes. |
| `snapshot-<number>.idx` | A complete saved index, with every change before that sequence number. |
| `wal-<number>.log` | A log segment starting at that sequence number. |
| `LOCK` | Locked by the operating system while the index is open. |

Do not change these files by hand. To back up a durable index, call `checkpoint()` and copy the directory while no change runs (or copy it while closed).

**Limitations:**
- **One process per directory.** A second open throws `std::runtime_error`.
- A checkpoint holds **changes** (not searches) while its snapshot is written.
- With `Synced`, a change reported as failed because its flush failed may still appear after a crash: its record may have reached the disk anyway (every database has this ambiguity).
- If a flush fails, or an abort record cannot be written, changes are refused until the next `checkpoint()`, which writes the true state.
- If a checkpoint fails (a full disk, say), the next attempt retries the snapshot; changes made meanwhile are safe in the log.
- After recovery, records match exactly; a parallel `add_batch` or a rebuild that ran alongside changes may leave a slightly different graph or codes, as can maintenance (which is not logged).

### 8.5 Low-level log access

For tools, and for logging a non-durable index yourself. Most applications never need these.

| API | What it does |
|---|---|
| `index.attach_log(WalWriter log)` | Logs every later change of a non-durable index to `log` (refused for durable indexes). The log's next sequence number must be at least 1. |
| `index.detach_log()` | Stops logging; returns the log. |
| `index.replay(records)` | Applies records in order, skipping aborted ones, as recovery does. No log may be attached (`std::logic_error`). Returns `true` if a failing **final** record was dropped as unfinished; a failure anywhere else throws `FormatError` (the log does not match the index). |
| `WalWriter::create(path, identity, first_lsn)`, `WalWriter::open(path, identity)` | Create a log segment, or open one for appending (cutting off a torn tail). |
| `writer.append(type, payload)`, `writer.sync()` | Append a record (returns its sequence number); force appended records onto the disk. A failed write is cut back; if that or a sync fails, the writer refuses further work (`broken()`). |
| `wal::read_segment(path)` | Reads a segment: `WalScan { std::uint64_t identity; std::uint64_t first_lsn; std::vector<WalRecord> records; std::uint64_t valid_end; std::uint64_t file_size; }`. `first_lsn` is the segment's first sequence number; `valid_end` is the byte offset just after the last valid record; `torn_tail()` is true when bytes after it were discarded. Each `WalRecord` is `{ std::uint64_t lsn; std::uint8_t type; std::vector<unsigned char> payload; }`. Throws `FormatError` for corruption. |
| `writer.next_lsn()`, `writer.size()`, `writer.broken()`, `writer.path()` | The sequence number the next record will get; the segment's size in bytes; whether the writer refuses work after a failure; its file. |
| `wal::sync_path(path)`, `wal::sync_directory(dir)`, `FileLock(path)` | Flush a file; flush a directory's entries (no-op on Windows); hold an exclusive lock on a file. |
| `wal_faults` | Switches that simulate disk failures, **for tests only**. |

---

## 9. Health, upkeep, rebuilds and long operations

`#include "health.h"`

### 9.1 Health reports

```cpp
HealthReport health(const HealthOptions& options = {}) const;   // HnswIndex, FlatIndex, DiskIndex
```

| `HealthOptions` field | Default | What it does |
|---|---|---|
| `recall_sample` | 100 | Stored vectors searched for to estimate recall (0: no estimate). Each costs one search and one exact scan, so this is the expensive part. |
| `k`, `ef` | 10, 64 | The recall estimated is recall@k at this beam width. |
| `seed` | 1 | Which vectors are sampled. |
| `codec_sample` | 500 | Vectors on which compression and IVF figures are measured (0: none). Needs the full vectors (not codes only). |
| `retrain_sample` | 0 | Further vectors on which the codec and IVF centroids are retrained in memory, to show what retraining would gain (0: no comparison). Costs a training. |

`HealthReport` fields, by area:

- **Storage:** `live` vectors, `slots`, `free_slots`.
- **Graph** (bottom level unless stated): `levels` (nodes per level against the number expected), `mean_degree`, `min_degree`, `underfull` (fewer than M neighbors), `no_incoming` (no live node links to it: fragile), `unreachable` (a search cannot reach it: it can never be found), `links_to_removed` (wasted work for searches), `stale_links`.
- **Compression:** `codec_error` (reconstruction error relative to the vectors' size; 0 is exact), `codec_error_retrained` (the same after retraining: **much lower means the data has drifted, and a rebuild would help**), `clipped` (SQ8: the share of values outside the trained range).
- **IVF:** `lists`, `lists_trained`, `largest_list`, `empty_lists`, `list_imbalance` (1 is perfectly balanced), `centroid_distance` and `centroid_distance_retrained`.
- **Quality:** `recall`, the estimated recall@k (none if not requested or codes only).
- **Churn:** `removals` and `updates` since the last complete upkeep; `churn()` is their sum over `live`.

**What to do with it:** `unreachable`, `links_to_removed` or high churn: run `upkeep()`. Retrained error much lower than current: run `rebuild()`. Low `recall`: raise `ef` when searching, or `M` and `ef_construction` when building.

### 9.2 Upkeep

| `UpkeepOptions` field | Default | What it does |
|---|---|---|
| `consolidate` | `true` | Replace links to removed vectors (and stale links) with the best of the node's other links and the removed vectors' live links. |
| `reconnect` | `true` | Give every unreachable vector an incoming link from a nearby reachable one. |
| `max_nodes` | 0 | Vectors repaired per call (0: no limit). Each repaired vector leaves a valid graph, so calling again continues. |

`UpkeepStats` reports `nodes_repaired`, `links_replaced`, `reconnected`, `unreachable` (still unreachable afterwards; counted only when complete) and `complete` (nothing stopped it; the churn counters then reset).

### 9.3 Rebuilds and compaction

`RebuildOptions { bool retrain = true; }`: retrain the codec (and IVF lists) on the current vectors. Without the full vectors (codes only) the codec is kept as it is.

`RebuildStats` reports `kept` vectors, `reclaimed` free slots, `caught_up` (IDs changed during the rebuild and copied in before the swap), `codec_retrained` and `lists_retrained`. `CompactStats` reports `kept` and `reclaimed`.

### 9.4 Long operations: `OperationOptions` and `Cancelled`

`add_batch`, `compact`, `rebuild`, `save` and `upkeep` accept:

```cpp
struct OperationOptions {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(std::size_t done, std::size_t total)> progress;
};
```

| Field | What it does |
|---|---|
| `deadline` | Give up when this moment passes. |
| `cancel` | Give up when another thread sets the flag to `true`. |
| `progress` | Called with (done, total): starting at (0, total), never decreasing, at most about 100 times, and ending at (total, total) on success. Calls never overlap but may come from worker threads. It **must not use the index**. If it throws, the operation is undone and the exception passed on. |

When a deadline passes or cancellation is requested, the operation is **undone** and throws `Cancelled` (a `std::runtime_error`): the index is as before, and a save leaves no temporary file. **Exception:** `upkeep()` does not throw; it stops early and keeps what it repaired (after at least one repair, for a deadline).

---

## 10. Errors at a glance

| Exception | Thrown for | State afterwards |
|---|---|---|
| `std::invalid_argument` | Invalid input: wrong dimension, NaN or infinity, a duplicate ID, metadata not fitting the schema, invalid settings, a filter on an unknown field (strict schema), a query part for a missing side | Unchanged |
| `std::logic_error` | Misuse: a change from inside an index operation on the same thread, a second concurrent rebuild, `checkpoint()` on a non-durable index, swapping a durable index's log, training a codes-only codec again | Unchanged |
| `std::runtime_error` | The outside world: a file that cannot be opened or written, a directory locked by another process, a broken log refusing changes | Unchanged (the failed change is not made) |
| `FormatError` (a `std::runtime_error`) | A damaged, truncated or foreign file or directory; a log that does not match its index | Nothing is loaded |
| `Cancelled` (a `std::runtime_error`) | A deadline or cancellation stopped a long operation | Undone |
| `std::bad_alloc` | Out of memory | Unchanged (the change is rolled back) |

Searches stopped by a deadline or cancellation do **not** throw: they return the best results so far and set `SearchStats::stopped_early`.

---

## 11. Limits at a glance

| Area | Limit |
|---|---|
| IDs | 64-bit unsigned integers; collections can also take string keys (opt-in `string_ids`), hashed to 64-bit ids with collisions refused |
| Vectors | `float` only (no half precision or integers); one dimension per index, from 1 to 16,777,216 (2²⁴); finite values only |
| Memory | In general about 4 × dim + 8 × M + 45 bytes per vector (2 × dim with half precision and codes only), plus metadata. Measured with M 16: 687 bytes at 128 dimensions (448 with float16 or bfloat16, 35% less) and 3,280 at 768 (1,753 with half precision, 47% less). Storage grows in blocks of up to 65,536 rows, but unused rows cost no resident memory |
| Metrics | L2 (squared), inner product, cosine; `DiskIndex`: L2 and cosine |
| Graph | `M` from 2 to 2048; HNSW search is approximate |
| Metadata | Integer, float, boolean, keyword and tag-set fields; dates as integers (milliseconds since 1970, `parse_date`); no geographic types; a field keeps its first type |
| Vector magnitudes | Any finite float is accepted. A distance that overflows becomes infinity (never NaN), and infinities sort last, ties in insertion order. Consequence: in an L2 or inner-product graph, a vector whose distances to every neighbor overflow (components near 3e38) ties them all at infinity, so graph search may not reach it; exact search (`Strategy::ForceExact`, `FlatIndex`) still finds it. Cosine normalizes, so it is unaffected. Real embeddings are many orders of magnitude below this. |
| Half precision | `F16` refuses values beyond ±65504 (use `BF16` for larger ones); queries stay float32, so any finite query works. The disk index (`DiskIndex`) keeps float32 vectors. Building a half-precision index costs a little more time at high dimensions (measured: 26% at 768, none at 128). |
| Filters | At most 256 levels deep; whole-value comparisons (no substring or full-text matching) |
| Text | No stemming, stop words or synonyms; 32-bit hashed terms |
| Concurrency | One change at a time per index or collection; searches in parallel; no changes from inside predicates |
| Durability | One process per directory; a checkpoint holds changes while writing; maintenance not logged |
| Disk index | Read-only; built with the vectors in memory (or partitioned from a file) |
| Hardware | CPU only (AVX2, AVX-512, NEON or plain code, chosen at run time); no GPU |
| Distribution | A library, not a server: no replication, sharding or network access |

---

## Appendix: complete signatures

Every public type, field and method exactly as declared in the headers: parameter types, names and default values, and every field's type and default. Generated from the compiler's view of the code, so it cannot drift from it; the sections above explain what each one does.

<!-- GENERATED by .github/scripts/gen_signatures.py from the headers: do not edit by hand -->

### Collections (collection.h)

**`Preset`** (enumeration)

```cpp
enum class Preset { Balanced, HighRecall, LowMemory };
```

**`Maintenance`** (enumeration)

```cpp
enum class Maintenance { Manual, Background };
```

**`MaintenancePolicy`**

```cpp
struct MaintenancePolicy {
    Maintenance mode = Maintenance::Manual;
    std::chrono::milliseconds interval{1000};
    std::chrono::milliseconds budget{250};
    std::uint64_t checkpoint_bytes = 64ull << 20;
    std::uint64_t checkpoint_records = 100000;
    std::chrono::seconds checkpoint_age{600};
    double health_check_churn = 0.05;
    double upkeep_churn = 0.2;
    std::size_t upkeep_nodes = 0;
    double rebuild_check_churn = 0.5;
    double rebuild_drift = 4.0;
};
```

**`MaintenanceReport`**

```cpp
struct MaintenanceReport {
    std::chrono::system_clock::time_point when{};
    std::chrono::microseconds elapsed{0};
    std::optional<HealthReport> health;
    bool upkept = false;
    UpkeepStats upkeep;
    bool rebuilt = false;
    RebuildStats rebuild;
    bool checkpointed = false;
    std::uint64_t checkpoint_lsn = 0;
    bool budget_exhausted = false;
    std::vector<std::string> notes;
    std::string error;
};
```

**`MaintenanceStatus`**

```cpp
struct MaintenanceStatus {
    std::uint64_t log_bytes = 0;
    std::uint64_t log_records = 0;
    std::chrono::seconds since_checkpoint{0};
    double churn_since_upkeep = 0;
    double churn_since_rebuild = 0;
    std::size_t rounds = 0;
};
```

**`TtlOptions`**

```cpp
struct TtlOptions {
    bool enabled = false;
    std::int64_t default_ms = 0;
};
```

**`CollectionOptions`**

```cpp
struct CollectionOptions {
    std::size_t dim = 0;
    Metric metric = Metric::Cosine;
    HnswParams hnsw;
    Schema schema = Schema::dynamic();
    bool text = true;
    Bm25Params bm25;
    bool sparse = false;
    bool store_text = true;
    DurableOptions durability;
    MaintenancePolicy maintenance;
    std::size_t ef = 0;
    static CollectionOptions preset(std::size_t dim, Preset preset = Preset::Balanced, std::size_t expected_size = 100000);
    TtlOptions ttl;
    bool string_ids = false;
};
```

**`GroupBy`**

```cpp
struct GroupBy {
    std::string field;
    std::size_t groups = 10;
    std::size_t per_group = 3;
    std::size_t max_candidates = 0;
};
```

**`CollectionGroup`**

```cpp
struct CollectionGroup {
    Value value;
    std::vector<CollectionHit> hits;
};
```

**`Recommend`**

```cpp
struct Recommend {
    std::vector<std::uint64_t> positive;
    std::vector<std::uint64_t> negative;
};
```

**`FacetRequest`**

```cpp
struct FacetRequest {
    std::string field;
    std::size_t limit = 10;
    std::vector<double> boundaries;
    bool distinct = false;
    void validate() const;
};
```

**`FacetValue`**

```cpp
struct FacetValue {
    Value value;
    std::size_t count = 0;
};
```

**`FacetBucket`**

```cpp
struct FacetBucket {
    std::optional<double> from;
    std::optional<double> to;
    std::size_t count = 0;
};
```

**`FacetResult`**

```cpp
struct FacetResult {
    std::string field;
    std::size_t total = 0;
    std::size_t missing = 0;
    std::vector<FacetValue> values;
    std::size_t other = 0;
    std::vector<FacetBucket> buckets;
    std::optional<std::size_t> distinct;
};
```

**`StoredInfo`**

```cpp
struct StoredInfo {
    StoredKind kind = StoredKind::Hnsw;
    bool durable = false;
    std::uint32_t version = 0;
    std::uint64_t bytes = 0;
    std::uint64_t identity = 0;
    std::vector<std::uint64_t> snapshots;
    std::vector<std::uint64_t> segments;
};
```

**`Document`**

```cpp
struct Document {
    std::vector<float> vector;
    std::string text;
    SparseVector sparse;
    Metadata metadata;
    std::optional<std::int64_t> expires_at;
};
```

**`DocumentRecord`**

```cpp
struct DocumentRecord {
    std::uint64_t id = 0;
    std::optional<std::vector<float>> vector;
    std::string text;
    std::optional<SparseVector> sparse;
    Metadata metadata;
    std::optional<std::string> key;
    std::optional<std::int64_t> expires_at;
};
```

**`CollectionScanOptions`**

```cpp
struct CollectionScanOptions {
    std::optional<Filter> filter;
    std::size_t limit = 100;
    std::optional<std::uint64_t> after;
    bool with_vectors = true;
    bool with_text = true;
    bool with_metadata = true;
};
```

**`CollectionPage`**

```cpp
struct CollectionPage {
    std::vector<DocumentRecord> documents;
    std::optional<std::uint64_t> next;
};
```

**`CollectionQuery`**

```cpp
struct CollectionQuery {
    std::vector<float> vector;
    std::string text;
    SparseVector sparse;
};
```

**`CollectionSearchOptions`**

```cpp
struct CollectionSearchOptions {
    std::optional<Filter> filter;
    std::function<bool(std::uint64_t)> predicate;
    Fusion fusion = Fusion::Rrf;
    double rrf_k = 60.0;
    double vector_weight = 1.0;
    double vector_weight = 1.0, text_weight = 1.0;
    double vector_weight = 1.0, text_weight = 1.0, sparse_weight = 1.0;
    std::size_t candidates = 0;
    std::size_t ef = 0;
    bool with_metadata = false;
    bool with_text = false;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    bool* stopped_early = nullptr;
    std::optional<ScoreModifiers> modifiers;
    std::optional<double> diversity;
};
```

**`CollectionHit`**

```cpp
struct CollectionHit {
    std::uint64_t id = 0;
    double score = 0;
    std::optional<std::size_t> vector_rank;
    std::optional<std::size_t> vector_rank, text_rank;
    std::optional<std::size_t> vector_rank, text_rank, sparse_rank;
    std::optional<float> vector_distance;
    std::optional<float> text_score;
    std::optional<float> text_score, sparse_score;
    std::optional<Metadata> metadata;
    std::optional<std::string> text;
    std::optional<double> base_score;
    std::optional<std::string> key;
};
```

**`Collection`**

```cpp
class Collection {
    explicit Collection(CollectionOptions options);
    static std::unique_ptr<Collection> open(const std::string& dir, CollectionOptions options, RecoveryStats* stats = nullptr);
    std::uint64_t checkpoint();
    bool durable() const;
    void add(std::uint64_t id, const Document& document);
    bool upsert(std::uint64_t id, const Document& document);
    bool remove(std::uint64_t id);
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::size_t remove_where(const Filter& filter);
    bool set_metadata(std::uint64_t id, const Metadata& metadata);
    std::size_t set_metadata_where(const Filter& filter, const Metadata& changes);
    bool contains(std::uint64_t id) const;
    std::size_t size() const;
    std::size_t count_where(const Filter& filter) const;
    void create_payload_index(std::string_view field);
    Schema schema() const;
    std::vector<std::string> payload_indexes() const;
    std::vector<FacetResult> facets(const std::vector<FacetRequest>& requests, const std::optional<Filter>& filter = std::nullopt) const;
    std::vector<FacetResult> facets_for(std::span<const std::uint64_t> ids, const std::vector<FacetRequest>& requests) const;
    std::optional<DocumentRecord> get(std::uint64_t id) const;
    CollectionPage scan(const CollectionScanOptions& options = {}) const;
    CollectionOptions options() const;
    std::vector<CollectionHit> search(const CollectionQuery& query, std::size_t k, const CollectionSearchOptions& options = {}) const;
    std::vector<CollectionGroup> search_groups(const CollectionQuery& query, const GroupBy& group_by, const CollectionSearchOptions& options = {}) const;
    std::vector<CollectionHit> recommend(const Recommend& examples, std::size_t k, const CollectionSearchOptions& options = {}) const;
    void save(const std::string& path) const;
    void save(std::ostream& out) const;
    static std::unique_ptr<Collection> load(const std::string& path);
    static std::unique_ptr<Collection> load(std::istream& in);
    bool replay(std::span<const WalRecord> records);
    std::size_t remove_expired();
    static std::uint64_t id_of(std::string_view key);
    bool set_metadata(std::string_view key, const Metadata& metadata);
    void set_ttl(const TtlOptions& ttl);
    void set_string_ids(bool on);
    void add(std::string_view key, const Document& document);
    bool upsert(std::string_view key, const Document& document);
    bool remove(std::string_view key);
    bool contains(std::string_view key) const;
    std::optional<DocumentRecord> get(std::string_view key) const;
    MaintenanceReport maintain(std::chrono::milliseconds budget = std::chrono::milliseconds(0));
    void set_maintenance(const MaintenancePolicy& policy);
    std::optional<MaintenanceReport> last_maintenance() const;
    MaintenanceStatus maintenance_status() const;
    const HnswIndex* dense_index() const;
    const SparseIndex* text_index() const;
    const SparseIndex* sparse_index() const;
};
```

### Metadata and filters (metadata.h, filter.h)

**`FieldType`** (enumeration)

```cpp
enum class FieldType { Int, Float, Bool, Keyword, Tags };
```

**`Value`**

```cpp
struct Value {
    FieldType type = FieldType::Int;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<std::string> tags;
    static Value of_int(std::int64_t v);
    static Value of_float(double v);
    static Value of_bool(bool v);
    static Value of_keyword(std::string v);
    static Value of_tags(std::vector<std::string> v);
    template <class T> static Value from(T&& v);
    friend bool operator==(const Value& lhs, const Value& rhs);
};
```

**`Metadata`**

```cpp
class Metadata {
    template <class T> Metadata& set(std::string name, T&& value);
    Metadata& set_date(std::string name, std::string_view iso);
    Metadata& set_tags(std::string name, std::vector<std::string> tags);
    Metadata& unset(std::string name);
    std::optional<Value> get(std::string_view name) const;
    bool empty() const;
    std::size_t size() const;
    const std::map<std::string, std::optional<Value>>& fields() const;
    friend bool operator==(const Metadata& a, const Metadata& b);
};
```

**`FieldSpec`**

```cpp
struct FieldSpec {
    std::string name;
    FieldType type;
    friend bool operator==(const FieldSpec&, const FieldSpec&);
};
```

**`SchemaMode`** (enumeration)

```cpp
enum class SchemaMode { Strict, Dynamic };
```

**`Schema`**

```cpp
class Schema {
    static Schema strict(std::vector<FieldSpec> fields);
    static Schema dynamic(std::vector<FieldSpec> fields = {});
    SchemaMode mode() const;
    const std::vector<FieldSpec>& fields() const;
};
```

**`CompareOp`** (enumeration)

```cpp
enum class CompareOp { Eq, Ne, Lt, Le, Gt, Ge };
```

**`Filter`**

```cpp
class Filter {
    Filter();
    static Filter all();
    template <class T> static Filter eq(std::string field, T&& v);
    template <class T> static Filter ne(std::string field, T&& v);
    template <class T> static Filter lt(std::string field, T&& v);
    template <class T> static Filter le(std::string field, T&& v);
    template <class T> static Filter gt(std::string field, T&& v);
    template <class T> static Filter ge(std::string field, T&& v);
    template <class T, class U> static Filter between(std::string field, T&& lo, U&& hi);
    template <class T> static Filter in(std::string field, const std::vector<T>& values);
    template <class T> static Filter in(std::string field, std::initializer_list<T> values);
    static Filter has_tag(std::string field, std::string tag);
    static Filter has_any_tag(std::string field, std::vector<std::string> tags);
    static Filter has_all_tags(std::string field, std::vector<std::string> tags);
    static Filter geo_radius(std::string lat_field, std::string lon_field, double lat, double lon, double meters);
    static Filter geo_box(std::string lat_field, std::string lon_field, double south, double west, double north, double east);
    static Filter exists(std::string field);
    friend Filter operator&&(const Filter& x, const Filter& y);
    friend Filter operator||(const Filter& x, const Filter& y);
    friend Filter operator!(const Filter& x);
    const Node& root() const;
    bool is_all() const;
};
```

**`Strategy`** (enumeration)

```cpp
enum class Strategy { Auto, ForceExact, ForceGraph, ForceAcorn };
```

**`PlanReason`** (enumeration)

```cpp
enum class PlanReason { NoFilter, Forced, MatchCount, MatchFraction, Graph, FlatIndex, Acorn, ExactFallback };
```

**`PlannerParams`**

```cpp
struct PlannerParams {
    std::size_t max_exact_matches = 2000;
    double max_exact_fraction = 0.01;
    std::size_t sample_size = 256;
    double max_ef_multiplier = 32.0;
    double acorn_below = 0.2;
    double acorn_ef_multiplier = 4.0;
    double fallback_factor = 1.0;
};
```

**`DecayShape`** (enumeration)

```cpp
enum class DecayShape { Exponential, Linear };
```

**`RecencyDecay`**

```cpp
struct RecencyDecay {
    std::string field;
    double origin = 0;
    double half_life = 0;
    DecayShape shape = DecayShape::Exponential;
    double missing = 1.0;
    double floor = 0.0;
};
```

**`FieldMultiplier`**

```cpp
struct FieldMultiplier {
    std::string field;
    double missing = 1.0;
    double min = 0.0;
    double max = std::numeric_limits<double>::infinity();
};
```

**`MatchBoost`**

```cpp
struct MatchBoost {
    Filter when;
    double bonus = 0.0;
};
```

**`ScoreModifiers`**

```cpp
struct ScoreModifiers {
    std::vector<RecencyDecay> decays;
    std::vector<FieldMultiplier> multipliers;
    std::vector<MatchBoost> boosts;
    std::size_t pool = 0;
    bool empty() const;
    std::size_t pool_for(std::size_t k) const;
    void validate() const;
    double apply(double base, const Metadata& md) const;
};
```

**`SearchStats`**

```cpp
struct SearchStats {
    Strategy strategy = Strategy::Auto;
    PlanReason reason = PlanReason::NoFilter;
    double selectivity = 1.0;
    std::size_t estimated_matches = 0;
    bool exact_count = false;
    std::size_t ef = 0;
    std::size_t nodes_visited = 0;
    std::size_t distance_computations = 0;
    bool stopped_early = false;
};
```

**`SearchOptions`**

```cpp
struct SearchOptions {
    std::size_t probes = 0;
    std::optional<Filter> filter;
    std::function<bool(std::uint64_t)> predicate;
    Strategy strategy = Strategy::Auto;
    std::optional<PlannerParams> planner;
    SearchStats* stats = nullptr;
    std::size_t range_expand_outside = 1;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    std::optional<ScoreModifiers> modifiers;
    std::optional<double> diversity;
};
```

**`OperationOptions`**

```cpp
struct OperationOptions {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(std::size_t done, std::size_t total)> progress;
};
```

**`Record`**

```cpp
struct Record {
    std::uint64_t id = 0;
    std::vector<float> vector;
    Metadata metadata;
};
```

**`ScanOptions`**

```cpp
struct ScanOptions {
    std::optional<Filter> filter;
    std::size_t limit = 100;
    std::optional<std::uint64_t> after;
    bool with_vectors = true;
    bool with_metadata = true;
};
```

**`Page`**

```cpp
struct Page {
    std::vector<Record> records;
    std::optional<std::uint64_t> next;
};
```

**`PlanDecision`**

```cpp
struct PlanDecision {
    Strategy strategy = Strategy::ForceGraph;
    PlanReason reason = PlanReason::Graph;
    double selectivity = 1.0;
    std::size_t estimated_matches = 0;
    bool exact_count = false;
    std::size_t ef = 0;
};
```

### Results and distances (search_result.h, distance.h)

**`Metric`** (enumeration)

```cpp
enum class Metric { L2, InnerProduct, Cosine };
```

**`SearchResult`**

```cpp
struct SearchResult {
    std::uint64_t id;
    float distance;
    float score = 0;
};
```

**`CompactStats`**

```cpp
struct CompactStats {
    std::size_t kept;
    std::size_t reclaimed;
};
```

### HnswIndex (hnsw_index.h, codec.h)

**`CodecKind`** (enumeration)

```cpp
enum class CodecKind { None, SQ8, Binary, PQ, F16, BF16, RaBitQ };
```

**`QuantizationParams`**

```cpp
struct QuantizationParams {
    CodecKind codec = CodecKind::None;
    std::size_t train_after = 1000;
    bool rerank = true;
    std::size_t rerank_factor = 4;
    std::size_t truncate_to = 0;
    bool keep_full_vectors = true;
    std::size_t pq_subspaces = 0;
};
```

**`HnswParams`**

```cpp
struct HnswParams {
    std::size_t M = 16;
    std::size_t ef_construction = 200;
    std::uint64_t seed = 42;
    bool repair_on_remove = true;
    QuantizationParams quantization;
};
```

**`HnswIndex`**

```cpp
class HnswIndex {
    HnswIndex(std::size_t dim, Metric metric, HnswParams params = {}, Schema schema = Schema::dynamic());
    void add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata = {});
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads = 0, const OperationOptions& options = {});
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::span<const Metadata> metadata, std::size_t threads = 0, const OperationOptions& options = {});
    HealthReport health(const HealthOptions& options = {}) const;
    UpkeepStats upkeep(const UpkeepOptions& upkeep_options = {}, const OperationOptions& options = {});
    void set_auto_compact(double fraction);
    double auto_compact() const;
    RebuildStats rebuild(const RebuildOptions& rebuild_options = {}, const OperationOptions& options = {});
    void attach_log(WalWriter log);
    std::optional<WalWriter> detach_log();
    bool replay(std::span<const WalRecord> records);
    static std::unique_ptr<HnswIndex> open_durable(const std::string& dir, std::size_t dim, Metric metric, const HnswParams& params = {}, Schema schema = Schema::dynamic(), const DurableOptions& options = {}, RecoveryStats* stats = nullptr);
    std::uint64_t checkpoint();
    bool durable() const;
    std::optional<std::vector<SearchResult>> search_by_id(std::uint64_t id, std::size_t k, std::size_t ef = 64, const SearchOptions& options = {}, bool include_self = false) const;
    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;
    void train_codec();
    bool codec_trained() const;
    bool has_vector_values() const;
    const VectorCodec* codec() const;
    std::optional<std::vector<std::uint8_t>> get_code(std::uint64_t id) const;
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::optional<Record> get(std::uint64_t id) const;
    Page scan(const ScanOptions& options = {}) const;
    std::size_t remove_where(const Filter& filter);
    std::size_t set_metadata_where(const Filter& filter, const Metadata& changes);
    std::size_t count_where(const Filter& filter) const;
    Schema schema() const;
    std::vector<std::string> payload_indexes() const;
    std::vector<FacetResult> facets(const std::vector<FacetRequest>& requests, const std::optional<Filter>& filter = std::nullopt) const;
    std::vector<FacetResult> facets_for(std::span<const std::uint64_t> ids, const std::vector<FacetRequest>& requests) const;
    bool remove(std::uint64_t id);
    bool update(std::uint64_t id, std::span<const float> vector);
    bool update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);
    bool upsert(std::uint64_t id, std::span<const float> vector);
    bool upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);
    bool contains(std::uint64_t id) const;
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t ef = 64) const;
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, std::size_t ef, const SearchOptions& options) const;
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k, std::size_t ef = 64, std::size_t threads = 0, const SearchOptions& options = {}) const;
    std::vector<SearchResult> search_range(std::span<const float> query, float radius, std::size_t max_results = kNoLimit, const SearchOptions& options = {}) const;
    bool set_metadata(std::uint64_t id, const Metadata& metadata);
    std::optional<Metadata> get_metadata(std::uint64_t id) const;
    void create_payload_index(std::string_view field);
    void set_planner_params(const PlannerParams& p);
    PlannerParams planner_params() const;
    const MetadataStore& metadata() const;
    std::size_t pooled_visited_lists() const;
    CompactStats compact(const OperationOptions& options = {});
    CompactStats reorder(const OperationOptions& options = {});
    void save(const std::string& path, const OperationOptions& options = {}) const;
    void save(std::ostream& out, const OperationOptions& options = {}) const;
    static std::unique_ptr<HnswIndex> load(const std::string& path);
    static std::unique_ptr<HnswIndex> load(std::istream& in);
    std::size_t size() const;
    std::size_t deleted_count() const;
    std::size_t capacity() const;
    std::size_t dim() const;
    std::size_t indexed_dim() const;
    Metric metric() const;
    const HnswParams& params() const;
    int max_level() const;
    NodeId entry_point() const;
    const Storage& storage() const;
};
```

### FlatIndex (flat_index.h)

**`IvfParams`**

```cpp
struct IvfParams {
    std::size_t lists = 0;
    std::size_t probes = 8;
    std::size_t train_after = 10000;
};
```

**`FlatIndex`**

```cpp
class FlatIndex {
    FlatIndex(std::size_t dim, Metric metric, Schema schema = Schema::dynamic(), QuantizationParams quantization = {}, IvfParams ivf = {});
    void add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata = {});
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t threads = 0, const OperationOptions& options = {});
    void add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors, std::span<const Metadata> metadata, std::size_t threads = 0, const OperationOptions& options = {});
    HealthReport health(const HealthOptions& options = {}) const;
    RebuildStats rebuild(const RebuildOptions& rebuild_options = {}, const OperationOptions& options = {});
    void attach_log(WalWriter log);
    std::optional<WalWriter> detach_log();
    bool replay(std::span<const WalRecord> records);
    static std::unique_ptr<FlatIndex> open_durable(const std::string& dir, std::size_t dim, Metric metric, Schema schema = Schema::dynamic(), const QuantizationParams& quantization = {}, const IvfParams& ivf = {}, const DurableOptions& options = {}, RecoveryStats* stats = nullptr);
    std::uint64_t checkpoint();
    bool durable() const;
    std::optional<std::vector<SearchResult>> search_by_id(std::uint64_t id, std::size_t k, const SearchOptions& options = {}, bool include_self = false) const;
    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;
    void train_codec();
    bool codec_trained() const;
    bool has_vector_values() const;
    const VectorCodec* codec() const;
    std::optional<std::vector<std::uint8_t>> get_code(std::uint64_t id) const;
    const QuantizationParams& quantization() const;
    void train_lists();
    bool lists_trained() const;
    std::optional<std::uint32_t> list_of(std::uint64_t id) const;
    std::vector<std::size_t> list_sizes() const;
    const IvfParams& ivf() const;
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::optional<Record> get(std::uint64_t id) const;
    Page scan(const ScanOptions& options = {}) const;
    std::size_t remove_where(const Filter& filter);
    std::size_t set_metadata_where(const Filter& filter, const Metadata& changes);
    std::size_t count_where(const Filter& filter) const;
    Schema schema() const;
    std::vector<std::string> payload_indexes() const;
    std::vector<FacetResult> facets(const std::vector<FacetRequest>& requests, const std::optional<Filter>& filter = std::nullopt) const;
    std::vector<FacetResult> facets_for(std::span<const std::uint64_t> ids, const std::vector<FacetRequest>& requests) const;
    bool remove(std::uint64_t id);
    bool update(std::uint64_t id, std::span<const float> vector);
    bool update(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);
    bool upsert(std::uint64_t id, std::span<const float> vector);
    bool upsert(std::uint64_t id, std::span<const float> vector, const Metadata& metadata);
    bool contains(std::uint64_t id) const;
    bool set_metadata(std::uint64_t id, const Metadata& metadata);
    std::optional<Metadata> get_metadata(std::uint64_t id) const;
    void create_payload_index(std::string_view field);
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, const SearchOptions& options) const;
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k, std::size_t threads = 0, const SearchOptions& options = {}) const;
    std::vector<SearchResult> search_range(std::span<const float> query, float radius, std::size_t max_results = kNoLimit, const SearchOptions& options = {}) const;
    CompactStats compact(const OperationOptions& options = {});
    void save(const std::string& path, const OperationOptions& options = {}) const;
    void save(std::ostream& out, const OperationOptions& options = {}) const;
    static std::unique_ptr<FlatIndex> load(const std::string& path);
    static std::unique_ptr<FlatIndex> load(std::istream& in);
    std::size_t size() const;
    std::size_t deleted_count() const;
    std::size_t capacity() const;
    std::size_t dim() const;
    std::size_t indexed_dim() const;
    Metric metric() const;
    const MetadataStore& metadata() const;
};
```

### DiskIndex (disk_index.h, vamana.h)

**`VamanaParams`**

```cpp
struct VamanaParams {
    std::size_t R = 64;
    std::size_t L = 100;
    float alpha = 1.2f;
    std::uint64_t seed = 1;
};
```

**`DiskVectorFormat`** (enumeration)

```cpp
enum class DiskVectorFormat { Float32, Float16, BFloat16 };
```

**`DiskIndexParams`**

```cpp
struct DiskIndexParams {
    VamanaParams graph;
    DiskVectorFormat vectors = DiskVectorFormat::Float32;
    std::size_t pq_subspaces = 0;
};
```

**`LargeBuildParams`**

```cpp
struct LargeBuildParams {
    DiskIndexParams index;
    std::size_t memory_budget = std::size_t(1) << 30;
    std::size_t replicas = 2;
    std::size_t partitions = 0;
};
```

**`LargeBuildStats`**

```cpp
struct LargeBuildStats {
    std::size_t partitions = 0;
    std::size_t largest_partition = 0;
    std::size_t merged_and_pruned = 0;
};
```

**`DiskOpenOptions`**

```cpp
struct DiskOpenOptions {
    std::size_t io_threads = 0;
    std::chrono::microseconds simulated_latency{0};
};
```

**`DiskSearchParams`**

```cpp
struct DiskSearchParams {
    std::size_t L = 100;
    std::size_t beam_width = 4;
};
```

**`DiskSearchStats`**

```cpp
struct DiskSearchStats {
    std::size_t nodes_read = 0;
    std::size_t blocks_read = 0;
    std::size_t rounds = 0;
    bool stopped_early = false;
};
```

**`DiskIndex`**

```cpp
class DiskIndex {
    static void build(const std::string& path, std::span<const std::uint64_t> ids, std::span<const float> vectors, std::size_t dim, Metric metric, const DiskIndexParams& params = {}, std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic());
    static void build_from_file(const std::string& path, const std::string& vectors_path, Metric metric, const LargeBuildParams& params = {}, std::span<const std::uint64_t> ids = {}, std::span<const Metadata> metadata = {}, Schema schema = Schema::dynamic(), LargeBuildStats* stats = nullptr);
    static std::unique_ptr<DiskIndex> open(const std::string& path, const DiskOpenOptions& options = {});
    std::size_t size() const;
    std::size_t dim() const;
    Metric metric() const;
    std::size_t max_degree() const;
    DiskVectorFormat vector_format() const;
    NodeId medoid() const;
    const MetadataStore& metadata() const;
    const PQCodec& codec() const;
    std::uint64_t id_of(NodeId node) const;
    std::optional<NodeId> node_of(std::uint64_t id) const;
    Node read_node(NodeId node) const;
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k, const DiskSearchParams& params = {}, const SearchOptions& options = {}, DiskSearchStats* stats = nullptr) const;
    std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries, std::size_t k, const DiskSearchParams& params = {}, std::size_t threads = 0, const SearchOptions& options = {}) const;
    std::optional<std::vector<SearchResult>> search_by_id(std::uint64_t id, std::size_t k, const DiskSearchParams& params = {}, const SearchOptions& options = {}, bool include_self = false) const;
    bool contains(std::uint64_t id) const;
    std::optional<std::vector<float>> get_vector(std::uint64_t id) const;
    std::optional<Record> get(std::uint64_t id) const;
    Page scan(const ScanOptions& options = {}) const;
    HealthReport health(const HealthOptions& options = {}) const;
    std::uint64_t blocks_read() const;
};
```

**`BlockFile`**

```cpp
class BlockFile {
    explicit BlockFile(const std::string& path);
    std::uint64_t size() const;
    void read(std::uint64_t offset, void* buffer, std::size_t n) const;
    void set_simulated_latency(std::chrono::microseconds latency);
};
```

### Sparse vectors, BM25 and hybrid search (sparse.h)

**`SparseVector`**

```cpp
struct SparseVector {
    std::vector<std::uint32_t> indices;
    std::vector<float> values;
    std::size_t size() const;
    bool empty() const;
    friend bool operator==(const SparseVector&, const SparseVector&);
};
```

**`SparseRecord`**

```cpp
struct SparseRecord {
    std::uint64_t id = 0;
    SparseVector vector;
    Metadata metadata;
};
```

**`SparseIndex`**

```cpp
class SparseIndex {
    explicit SparseIndex(Schema schema = Schema::dynamic());
    void add(std::uint64_t id, const SparseVector& vector, const Metadata& metadata = {});
    bool remove(std::uint64_t id);
    std::size_t remove_batch(std::span<const std::uint64_t> ids);
    std::size_t remove_where(const Filter& filter);
    bool update(std::uint64_t id, const SparseVector& vector);
    bool update(std::uint64_t id, const SparseVector& vector, const Metadata& metadata);
    bool upsert(std::uint64_t id, const SparseVector& vector);
    bool upsert(std::uint64_t id, const SparseVector& vector, const Metadata& metadata);
    bool set_metadata(std::uint64_t id, const Metadata& metadata);
    bool contains(std::uint64_t id) const;
    std::size_t size() const;
    std::optional<SparseVector> get_vector(std::uint64_t id) const;
    std::optional<Metadata> get_metadata(std::uint64_t id) const;
    std::vector<SparseRecord> records() const;
    std::vector<SearchResult> search(const SparseVector& query, std::size_t k, const SearchOptions& options = {}) const;
    std::size_t document_frequency(std::uint32_t term) const;
    std::size_t terms() const;
    std::size_t postings() const;
    std::size_t dead_postings() const;
    std::size_t slots() const;
    Schema schema() const;
    std::vector<std::string> payload_indexes() const;
    void compact();
    void save(const std::string& path) const;
    void save(std::ostream& out) const;
    static std::unique_ptr<SparseIndex> load(const std::string& path);
    static std::unique_ptr<SparseIndex> load(std::istream& in);
    static std::unique_ptr<SparseIndex> open_durable(const std::string& dir, Schema schema = Schema::dynamic(), const DurableOptions& options = {}, RecoveryStats* stats = nullptr);
    std::uint64_t checkpoint();
    bool replay(std::span<const WalRecord> records);
};
```

**`Bm25Params`**

```cpp
struct Bm25Params {
    double k1 = 1.2;
    double b = 0.75;
    double avg_doc_length = 100.0;
    bool unicode = false;
    bool stop_words = false;
    bool stem = false;
};
```

**`Bm25`**

```cpp
class Bm25 {
    explicit Bm25(Bm25Params params = {});
    static std::vector<std::string> tokenize(std::string_view text);
    std::vector<std::string> analyze(std::string_view text) const;
    static std::uint32_t term_id(std::string_view token);
    void fit(std::span<const std::string_view> documents);
    SparseVector encode_document(std::string_view text) const;
    SparseVector encode_query(std::string_view text, const SparseIndex& index) const;
    const Bm25Params& params() const;
};
```

**`Fusion`** (enumeration)

```cpp
enum class Fusion { Rrf, Weighted };
```

**`HybridOptions`**

```cpp
struct HybridOptions {
    Fusion fusion = Fusion::Rrf;
    double rrf_k = 60.0;
    double dense_weight = 1.0;
    double sparse_weight = 1.0;
    std::size_t dense_k = 0;
    std::size_t sparse_k = 0;
    std::size_t dense_ef = 0;
    std::optional<Filter> filter;
    SearchOptions dense_options;
    SearchOptions sparse_options;
    std::optional<ScoreModifiers> modifiers;
};
```

**`HybridResult`**

```cpp
struct HybridResult {
    std::uint64_t id = 0;
    double score = 0;
    std::optional<std::size_t> dense_rank;
    std::optional<std::size_t> sparse_rank;
    std::optional<float> dense_distance;
    std::optional<float> sparse_score;
    std::optional<double> base_score;
};
```

### Health, upkeep and rebuilds (health.h)

**`HealthOptions`**

```cpp
struct HealthOptions {
    std::size_t recall_sample = 100;
    std::size_t k = 10;
    std::size_t ef = 64;
    std::uint64_t seed = 1;
    std::size_t codec_sample = 500;
    std::size_t retrain_sample = 0;
};
```

**`LevelHealth`**

```cpp
struct LevelHealth {
    std::size_t nodes = 0;
    double expected = 0;
};
```

**`HealthReport`**

```cpp
struct HealthReport {
    std::size_t live = 0;
    std::size_t slots = 0;
    std::size_t free_slots = 0;
    std::vector<LevelHealth> levels;
    double mean_degree = 0;
    std::size_t min_degree = 0;
    std::size_t underfull = 0;
    std::size_t no_incoming = 0;
    std::size_t unreachable = 0;
    std::size_t links_to_removed = 0;
    std::size_t stale_links = 0;
    std::optional<double> codec_error;
    std::optional<double> codec_error_retrained;
    std::optional<double> clipped;
    std::size_t lists = 0;
    bool lists_trained = false;
    std::size_t largest_list = 0;
    std::size_t empty_lists = 0;
    double list_imbalance = 0;
    std::optional<double> centroid_distance;
    std::optional<double> centroid_distance_retrained;
    std::optional<double> recall;
    std::size_t removals = 0;
    std::size_t updates = 0;
    double churn() const;
};
```

**`UpkeepOptions`**

```cpp
struct UpkeepOptions {
    bool consolidate = true;
    bool reconnect = true;
    std::size_t max_nodes = 0;
};
```

**`UpkeepStats`**

```cpp
struct UpkeepStats {
    std::size_t nodes_repaired = 0;
    std::size_t links_replaced = 0;
    std::size_t reconnected = 0;
    std::size_t unreachable = 0;
    bool complete = false;
};
```

**`RebuildOptions`**

```cpp
struct RebuildOptions {
    bool retrain = true;
};
```

**`RebuildStats`**

```cpp
struct RebuildStats {
    std::size_t kept = 0;
    std::size_t reclaimed = 0;
    std::size_t caught_up = 0;
    bool codec_retrained = false;
    bool lists_retrained = false;
};
```

### Durability (durable.h, wal.h)

**`Durability`** (enumeration)

```cpp
enum class Durability { Buffered, Grouped, Synced };
```

**`DurableOptions`**

```cpp
struct DurableOptions {
    std::size_t keep_snapshots = 2;
    Durability durability = Durability::Grouped;
    std::chrono::milliseconds group_interval{10};
};
```

**`RecoveryStats`**

```cpp
struct RecoveryStats {
    bool created = false;
    std::uint64_t snapshot_lsn = 0;
    std::size_t fallbacks = 0;
    std::size_t segments = 0;
    std::size_t replayed = 0;
    std::uint64_t torn_bytes = 0;
    bool dropped_unfinished = false;
};
```

**`WalRecord`**

```cpp
struct WalRecord {
    std::uint64_t lsn = 0;
    std::uint8_t type = 0;
    std::vector<unsigned char> payload;
};
```

**`WalScan`**

```cpp
struct WalScan {
    std::uint64_t identity = 0;
    std::uint64_t first_lsn = 0;
    std::vector<WalRecord> records;
    std::uint64_t valid_end = 0;
    std::uint64_t file_size = 0;
    bool torn_tail() const;
};
```

**`WalWriter`**

```cpp
class WalWriter {
    static WalWriter create(const std::string& path, std::uint64_t identity, std::uint64_t first_lsn);
    static WalWriter open(const std::string& path, std::uint64_t identity);
    WalWriter(WalWriter&& other) noexcept;
    std::uint64_t append(std::uint8_t type, std::span<const unsigned char> payload);
    void sync();
    std::uint64_t next_lsn() const;
    std::uint64_t size() const;
    bool broken() const;
    const std::string& path() const;
};
```

**`FileLock`**

```cpp
class FileLock {
    explicit FileLock(const std::string& path);
    FileLock(FileLock&& o) noexcept;
};
```

### Free functions

```cpp
template <class DenseIndex> std::vector<HybridResult> hybrid_search(const DenseIndex& dense, std::span<const float> dense_query, const SparseIndex& sparse, const SparseVector& sparse_query, std::size_t k, const HybridOptions& options = {});
std::vector<HybridResult> fuse(std::span<const SearchResult> dense, std::span<const SearchResult> sparse, std::size_t k, const HybridOptions& options = {});
double sparse_dot(const SparseVector& a, const SparseVector& b);
inline float l2_radius(float euclidean_radius);
void write_vectors_file(const std::string& path, std::span<const float> vectors, std::size_t dim);
WalScan wal::read_segment(const std::string& path);
void wal::sync_path(const std::string& path);
void wal::sync_directory(const std::string& dir);
PlanDecision plan_search(std::size_t capacity, std::size_t live, const Eligibility& eligible, std::optional<std::size_t> exact_matches, std::size_t ef, Strategy forced, const PlannerParams& params);
void validate(const PlannerParams& p);
inline void validate(const IvfParams& p);
inline void validate(const QuantizationParams& q);
void validate(const VamanaParams& p);
std::int64_t parse_date(std::string_view iso);
std::string format_date(std::int64_t ms);
std::int64_t now_ms();
inline double relevance_from_distance(float distance, Metric metric);
template <class MetadataOf, class Relevance> std::vector<SearchResult> rescored(std::vector<SearchResult> found, std::size_t k, const ScoreModifiers& m, MetadataOf&& metadata_of, Relevance&& relevance);
inline std::vector<std::size_t> mmr_order(const std::vector<double>& relevance, const std::vector<std::optional<std::vector<float>>>& vectors, std::size_t k, double lambda);
inline std::uint16_t float_to_f16(float f) noexcept;
inline float f16_to_float(std::uint16_t h) noexcept;
inline std::uint16_t float_to_bf16(float f) noexcept;
inline float bf16_to_float(std::uint16_t h) noexcept;
StoredInfo identify(const std::string& path);
const char* stored_kind_name(StoredKind kind);
std::string stem_english(std::string_view word);
std::vector<std::string> tokenize_unicode(std::string_view text);
bool is_english_stop_word(std::string_view token);
```

<!-- END GENERATED -->