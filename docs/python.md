# Python

> The package is called **hnsw-lite-lib** on PyPI and **hnsw_lite_lib** in Python: the name `hnsw-lite` belongs to an unrelated project.

hnsw-lite's Python package: `pip install hnsw-lite-lib`, then `import hnsw_lite_lib`. It is the C++ library with NumPy and pandas integration. Searches and writes run in C++ without holding Python's GIL, so several Python threads can search at once.

## Contents

- [Installing](#installing)
- [Quick start](#quick-start)
- [Collection](#collection)
- [pandas](#pandas)
- [HnswIndex and FlatIndex](#hnswindex-and-flatindex)
- [Coming from hnswlib](#coming-from-hnswlib)
- [Arrow, Parquet and importers](#arrow-parquet-and-importers)
- [Filters](#filters)
- [How Python values are converted](#how-python-values-are-converted)
- [Threads, errors and limits](#threads-errors-and-limits)
- [Releasing](#releasing)

## Installing

```bash
pip install hnsw-lite-lib                # or: pip install "hnsw-lite-lib[pandas]"
```

Prebuilt wheels cover CPython 3.9 to 3.13 on Linux (x86-64 and ARM64, glibc 2.28 or later: RHEL 8, Ubuntu 18.10, Debian 10 and newer), macOS (Apple Silicon, from macOS 11; Intel Macs build from source) and Windows (x86-64). Each wheel picks the fastest SIMD instructions the CPU has (AVX2, AVX-512 or NEON) when it loads. Elsewhere (Alpine Linux, 32-bit systems, PyPy), pip builds from the source distribution, which needs a C++20 compiler and CMake 3.20 or later.

From the repository: `pip install .` (or `pip install ".[test]"`, then `python -m pytest`, for the tests). NumPy is required; pandas is optional (only `upsert_dataframe`, `to_pandas` and `as_dataframe=True` use it).

## Quick start

```python
import numpy as np
import hnsw_lite_lib as hl

docs = hl.Collection(dim=384, path="docs.db")      # durable: created, or recovered after a crash
docs.upsert(1, vector=embedding, text="The quick brown fox", metadata={"lang": "en", "year": 2024, "tags": ["animals"]})

hits = docs.search(vector=query_embedding, text="fox", k=10, filter={"lang": "en"})
for hit in hits:
   print(hit["id"], hit["score"], hit["vector_rank"], hit["text_rank"])

docs.close()                                        # or use: with hl.Collection(...) as docs:
```

## Collection

The recommended object: documents with any of a dense vector, text (scored with BM25) and a learned sparse vector, plus metadata, under one ID space. It is the C++ [`Collection`](developer-reference.md#2-collection-the-recommended-starting-point); every rule and limit described there applies.

### Creating

```python
hl.Collection(dim=0, *, path=None, metric="cosine", preset="balanced", expected_size=100_000, text=True, sparse=False,
              store_text=True, schema=None, strict=False, durability="grouped", maintenance=None, ef=None,
              bm25_avg_doc_length=None, ttl=False, default_ttl=None, string_ids=False, stem=False,
              stop_words=False, unicode=False, options=None)
```

Text analysis for the text side (off by default, fixed at creation, saved with the collection): `hl.Collection(text=True, stem=True, stop_words=True, unicode=True)`. `stem` makes `runs` find `running` (Snowball English), `stop_words` ignores `the`, `and` and the like, and `unicode` lowercases beyond ASCII so `école` finds `École`. Reopening a durable collection needs the same settings.

| Argument | Type | What it does |
|---|---|---|
| `dim` | `int` | Vector length; 0 for a collection without vectors (text and/or sparse only). |
| `path` | `str` or `None` | A directory: the collection is **durable** (created, or recovered after a crash). `None`: in memory. |
| `metric` | `"cosine"`, `"l2"` (or `"euclidean"`), `"ip"` (or `"inner_product"`, `"dot"`), or `hl.Metric` | Distance of the vectors. |
| `preset` | `"balanced"`, `"high_recall"`, `"low_memory"`, or `hl.Preset` | Graph, compression and search settings (see [presets](developer-reference.md#22-presets)). Presets turn on background maintenance. |
| `expected_size` | `int` | About how many documents; from a million the presets use more links. |
| `text`, `sparse` | `bool` | Whether the collection has a BM25 text side and a learned sparse side. |
| `store_text` | `bool` | Keep each document's text, for `get()` and search results. |
| `schema` | `dict` or `None` | Field types: `{"year": int, "lang": "keyword", "tags": "tags"}`. Types: `int`, `float`, `bool`, `str` (or `"keyword"`), `list` (or `"tags"`), or `hl.FieldType`. |
| `strict` | `bool` | Only the schema's fields are allowed. |
| `durability` | `"buffered"`, `"grouped"`, `"synced"`, or `hl.Durability` | When the log reaches the disk (see [durability modes](developer-reference.md#81-durableoptions-and-durability)). |
| `maintenance` | `"manual"`, `"background"`, or `None` (the preset's choice) | Whether a background thread maintains the collection. |
| `ef` | `int` or `None` | Default search beam of the vectors (higher: better recall, slower). |
| `bm25_avg_doc_length` | `float` or `None` | Average document length in words, for BM25; set it near your corpus' mean. |
| `ttl`, `default_ttl` | `bool`; seconds or a `timedelta` | Opt-in expiry: documents expire at their `expires_at` (or after `default_ttl`); expired ones disappear from results at once. |
| `string_ids` | `bool` | Opt-in string keys (names, UUIDs) wherever an id is accepted. |
| `options` | `hl.CollectionOptions` or `None` | Every setting at once, instead of the arguments above (`hl.CollectionOptions.preset(dim, hl.Preset.BALANCED)` is a starting point). |

`hl.Collection.load(path, ttl=False, default_ttl=None, string_ids=False)` reads a file written by `save(path)`. Maintenance starts manual; expiry and string keys come back as saved (the arguments switch them on, never off).

### Methods

| Method | What it does |
|---|---|
| `add(id, vector=None, text=None, sparse=None, metadata=None, expires_at=None)` | Adds a document; `ValueError` if the ID exists or a part is invalid. |
| `upsert(id, vector=None, text=None, sparse=None, metadata=None, expires_at=None)` → `bool` | Adds, or replaces the **whole** document (parts left out are removed); `True` if added. |
| `upsert_many(ids, vectors=None, texts=None, sparse=None, metadata=None, expires_at=None)` → `int` | Many documents: `ids` (sequence or array), `vectors` (2-D array, one row per ID), and lists of texts, sparse vectors and metadata dicts, each optional, one entry per ID. Returns how many were added. |
| `upsert_dataframe(df, ...)` → `int` | See [pandas](#pandas). |
| `remove(ids)` → `int` | Removes one ID or many; how many were stored. |
| `remove_where(filter)`, `count_where(filter)` → `int` | By metadata. |
| `set_metadata(id, metadata)` → `bool` | Changes only the given fields; `None` clears a field. With string keys, `id` may be a key (only that exact key's document changes). |
| `set_metadata_where(filter, changes)` → `int` | The same for every match. |
| `get(id)` → `dict` or `None` | `{"id", "vector" (NumPy array or None), "text", "sparse" (dict or None), "metadata"}`. |
| `scan(*, filter=None, limit=100, after=None, with_vectors=True, with_text=True, with_metadata=True)` → `(list, next)` | One page of documents (dicts as from `get`), in ascending ID order; pass `next` as `after` for the following page, `None` on the last one. Changes between pages never repeat or skip a document that stays stored; expired documents are left out. |
| `documents(*, filter=None, batch=1000, ...)` → iterator | Every document (or those matching `filter`), in ascending ID order, read `batch` at a time so memory stays bounded. Same `with_*` switches as `scan`. |
| `to_pandas(*, filter=None, with_vectors=True, with_text=True, with_metadata=True, batch=10000)` → `DataFrame` | The documents as a DataFrame in the shape `upsert_dataframe` reads (see [pandas](#pandas)). |
| `facets(fields, *, filter=None, ids=None)` → `dict` | How many documents hold each value of metadata fields, over everything, those matching `filter`, or those with these `ids` (such as search hits). Each field is a name, or `{"field", "limit": 10, "ranges": [boundaries], "distinct": bool}`. Returns `{field: {"total", "missing", "values": [(value, count)], "other", "buckets": [{"from", "to", "count"}], "distinct"}}`. Expired documents never count. `HnswIndex` and `FlatIndex` have it too. |
| `create_payload_index(field)`, `payload_indexes()`, `schema()` | Exact counts and posting lists for a keyword, bool or tag field (fast facets and selective filters; logged when durable); the indexed fields; the fields and their types (`{name: "int" | "float" | "bool" | "keyword" | "tags"}`). |
| `search(vector=None, text=None, sparse=None, *, k=10, filter=None, ...)` | Below. |
| `search_groups(vector=None, text=None, sparse=None, *, group_by, groups=10, per_group=3, max_candidates=0, ...)` → `list` | Group-by: the best values of the metadata field `group_by`, each with its best documents, ranked as `search()` ranks them (the same query and options). Returns `[{"value": ..., "hits": [hit, ...]}, ...]`, hits shaped as `search()` returns them. A group short of `per_group` is filled by a search among its own documents: call `create_payload_index(group_by)` once so that reads only them. |
| `recommend(positive, negative=(), *, k=10, filter=None, ef=0, with_metadata=False, with_text=False, timeout=None)` → `list` | Documents like the positive examples and unlike the negative ones (IDs, or keys with `string_ids`), by their vectors; the examples are never returned. Hits shaped as `search()` returns them. |
| `checkpoint()` → `int` | Durable collections: a snapshot. |
| `maintain(budget_ms=0)` → `dict` | Runs the maintenance that is due; returns what it did (`upkept`, `rebuilt`, `checkpointed`, `notes`, ...). |
| `maintenance_status()` → `dict` | `log_bytes`, `log_records`, `since_checkpoint_s`, `churn_since_upkeep`, `churn_since_rebuild`, `rounds`. |
| `set_maintenance(mode="background", **settings)` | Changes the policy: `interval_ms`, `budget_ms`, `checkpoint_records`, `checkpoint_bytes`, `checkpoint_age_s`, `health_check_churn`, `upkeep_churn`, `upkeep_nodes`, `rebuild_check_churn`, `rebuild_drift`. |
| `save(path)`, `close()`, `len(c)`, `id in c`, `c.options`, `c.durable` | Files, lifetime, size, membership, settings. |

**Close durable collections** with `close()` or a `with` statement: closing flushes the log and releases the directory's lock (another process, or the same one, can then open it). A closed collection raises `ValueError` when used.

### Searching

```python
docs.search(vector=None, text=None, sparse=None, *, k=10, filter=None, fusion="rrf", rrf_k=60.0,
            vector_weight=1.0, text_weight=1.0, sparse_weight=1.0, candidates=0, ef=0,
            with_metadata=False, with_text=False, timeout=None, as_dataframe=False, modifiers=None,
            diversity=None, rerank=None, rerank_candidates=0, rerank_query=None)
```

Give at least one of `vector`, `text` and `sparse`. With several, each side's ranking is fused: Reciprocal Rank Fusion (`fusion="rrf"`) or scores scaled to [0, 1] (`"weighted"`), with a weight per side (see [fusion](developer-reference.md#27-searching)). `filter` is a [`Filter`](#filters) or a dict. `timeout` (seconds) stops the search early: the hits found so far are returned and `docs.last_search_stopped_early` is set to `True`.

Each hit is a dict: `id`, `score` (higher is better), `vector_rank`, `text_rank`, `sparse_rank` (1-based, or `None` if that side did not find it), `vector_distance`, `text_score`, `sparse_score`, and `metadata` and `text` when asked for.

### Reranking

A first search is fast but coarse; a cross-encoder reads the query and each candidate together and ranks better, but is too slow for a whole collection. `search` combines them: it fetches `rerank_candidates` hits (default 4k), lets the model re-score them by their text, and returns the k best of its scores.

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
hits = col.search(text="how do I reset my password", k=10, rerank=model, rerank_candidates=50)
```

`rerank` is either an object with `predict(pairs)` taking `(query, text)` pairs (sentence-transformers' `CrossEncoder`, and most rerankers), or a function `f(query, texts) -> scores`. The query is `text`, or `rerank_query` for a vector-only search. Hits keep their first-stage `score` and gain `rerank_score`, ordered by it (ties keep the first-stage order); documents without text are scored as an empty string. A reranker returning the wrong number of scores, or one that is not a finite number, raises `ValueError` rather than mis-ranking silently; the model's own errors propagate.

## Dates and score modifiers

Dates are stored as milliseconds since 1970 (UTC). In metadata and dict filters, `datetime`, `date`, pandas `Timestamp` and NumPy `datetime64` values convert automatically (naive datetimes are taken as UTC); `hl.to_ms(x)` converts explicitly (for `hl.Filter` functions), and `hl.parse_date`, `hl.format_date` and `hl.now_ms` are there too:

```python
docs.upsert(1, vector=v, metadata={"published": datetime.date(2024, 5, 17)})
docs.count_where({"published": datetime.date(2024, 5, 17)})
docs.count_where(hl.Filter.ge("published", hl.to_ms("2024-01-01")))
```

**Score modifiers (opt-in)** re-rank results by metadata, in C++ (no Python callback per result). Pass `modifiers=` to `Collection.search`, `HnswIndex.search` or `FlatIndex.search`:

```python
hits = docs.search(vector=q, text="pricing", k=10, modifiers={
    "decay": {"field": "published", "half_life_days": 30},        # recency; origin defaults to now
    "multiply": "provenance_confidence",                           # times a field (0 to 1)
    "boost": [({"edge_type": "verified_decision"}, 0.5),           # + 0.5 when it matches
              (hl.Filter.has_tag("tags", "stale"), -0.3)],          # a negative bonus demotes (tags: has_tag)
    "pool": 100,                                                   # candidates re-scored (default max(4k, 20))
})
```

| Key | Value |
|---|---|
| `decay` | A dict, or a list: `field`, a half-life (`half_life` in the field's units, or `half_life_seconds`, `_minutes`, `_hours`, `_days` for dates), `origin` (default now; a date, ISO string or number), `shape` (`"exponential"` or `"linear"`), `missing` (1.0), `floor` (0.0) |
| `multiply` | A field name, a dict (`field`, `missing`, `min`, `max`), or a list |
| `boost` | `(filter, bonus)` pairs or `{"when": filter, "bonus": x}` dicts (filters as `Filter` or dict), one or a list |
| `pool` | Candidates re-scored before the best k are kept |

The score is (base × decays × multipliers) + bonuses. Collection hits then carry `base_score` and `score`; index searches return `(ids, distances, scores)`, ordered by score. See the [developer reference](developer-reference.md#46-searching) for the exact formulas.

## Expiry, string keys and diversity

All opt-in:

```python
docs = hl.Collection(dim=384, ttl=True, string_ids=True)             # or default_ttl=3600 (seconds) / a timedelta
docs.upsert("doc-42", vector=v, expires_at="2025-01-01")             # any date form; a string key
docs.get("doc-42")                                                    # {..., "key": "doc-42", "expires_at": ...}
docs.remove_expired()                                                 # frees expired documents (already hidden)
hits = docs.search(vector=q, k=10, diversity=0.5)                     # MMR: 1 keeps the ranking, lower = more varied
ids, distances, scores = index.search(q, k=10, diversity=0.5)         # indexes too
```

`upsert_many` and `upsert_dataframe` take string keys (an `object` id column) and expiry times (`expires_at=` a list, `expires_column=`). Hits carry `key`; `hl.Collection.id_of(key)` gives a key's numeric id. The metadata names `_expires_at` and `_key` are reserved. `ttl` and `string_ids` are saved with the collection: a reopened or loaded collection keeps them (arguments switch a feature on, never off by omission).

## pandas

```python
docs.upsert_dataframe(df, *, id_column=None, vector_column=None, text_column=None, sparse_column=None, metadata_columns=None,
                      expires_column=None)
```

Upserts every row of a DataFrame:

- **IDs** come from `id_column`, or from the index if it is `None`. They must be non-negative integers.
- **Vectors** come from `vector_column`, whose cells hold arrays or lists (one vector per row), as embedding pipelines usually produce.
- **Text** comes from `text_column`; a missing value means no text.
- **Metadata** comes from `metadata_columns`, or by default **every other column**. Missing values (`NaN`, `None`, `pd.NA`, `NaT`) are left out of that row's metadata rather than stored. NumPy scalars become Python values, and cells holding lists become tag sets.

```python
results = docs.search(vector=q, text="fox", k=20, filter={"lang": "en"}, as_dataframe=True)
```

`as_dataframe=True` returns a DataFrame: `id`, `score`, `vector_rank`, `text_rank`, `sparse_rank` (pandas' nullable `Int64`, so missing ranks show as `<NA>`), `vector_distance`, `text_score`, `sparse_score`, optionally `text`, then **one column per metadata field**. A metadata field whose name collides with a result column (such as `score`) is renamed `metadata_score`.

The other way, `docs.to_pandas()` returns the documents as a DataFrame: an `id` column; `key` and `expires_at` (milliseconds since 1970 UTC) when the collection uses them; `vector` (one NumPy array per cell) and `sparse` (a dict per cell) for the sides the collection has; `text`; then one column per metadata field, missing values as `NaN` or `None` (a metadata field named like one of those columns gets a `metadata_` prefix). It is the shape `upsert_dataframe` reads, so a collection can be exported and loaded back:

```python
df = docs.to_pandas(filter={"lang": "en"})
copy.upsert_dataframe(df, id_column="id", vector_column="vector", text_column="text", metadata_columns=["year", "lang"])
```

It reads the collection `batch` documents per call (10,000 by default), then builds the DataFrame; for a collection too large to hold twice, iterate with `docs.documents()` instead.

**What a file holds.** `hl.identify(path)` reads only headers: `{"kind": "hnsw" | "flat" | "sparse" | "collection" | "disk", "durable", "version", "bytes"}`, plus `"identity"`, `"snapshots"` and `"segments"` for a durable directory. The `hnsw-lite` program does more (see the [user guide](user-guide.md)).

## HnswIndex and FlatIndex

For plain vector indexes with metadata, in the style of hnswlib and FAISS:

```python
index = hl.HnswIndex(dim=128, metric=hl.Metric.L2)            # or hl.FlatIndex(dim, metric) for exact search
index.add(np.arange(n), vectors, metadata=[{"lang": "en"}] * n, threads=0)
ids, distances = index.search(query, k=10, ef=64, filter=hl.Filter.eq("lang", "en"))
I, D = index.search_batch(queries, k=10, ef=64)                # shape (len(queries), 10)
```

| Method | What it does |
|---|---|
| `HnswIndex(dim, metric=Metric.COSINE, params=HnswParams(), schema=None, strict=False)` | `params` has `M`, `ef_construction`, `seed`, `repair_on_remove` and `quantization` (`QuantizationParams`: `codec`, `train_after`, `rerank`, `rerank_factor`, `truncate_to`, `keep_full_vectors`, `pq_subspaces`). |
| `FlatIndex(dim, metric=Metric.COSINE, schema=None, strict=False, quantization=QuantizationParams(), ivf=IvfParams())` | `ivf` has `lists`, `probes`, `train_after`. |
| `add(ids, vectors, metadata=None, threads=0)` | One vector (an int ID and a 1-D vector) or many (IDs and a 2-D array). Everything is checked first: on any error nothing is added. Built on `threads` threads (0: all cores). |
| `search(query, k=10, ef=64, filter=None, strategy=Strategy.AUTO)` | `(ids, distances)`: `uint64` and `float32` arrays, closest first. `FlatIndex` has no `ef` or `strategy`. |
| `search_batch(queries, k=10, ef=64, filter=None, threads=0)` | `(ids, distances)` of shape `(n, k)`. Where fewer than `k` results exist, rows are **padded** with `hl.MISSING_ID` (2⁶⁴−1) and `inf`. |
| `search_by_id(id, k=10, ef=64, filter=None)` | Like `search`, with the stored vector; the record itself is left out; `None` if not stored. |
| `upsert`, `update`, `remove`, `remove_where`, `count_where`, `set_metadata`, `set_metadata_where`, `get_vector`, `get_metadata` | As for collections. |
| `reorder()` (HnswIndex) | Opt-in: renumbers the graph so neighbors are stored together; same results, about 24% faster searches on clustered data. Call it after building or bulk loading (it blocks while it runs). |
| `create_payload_index(field)`, `facets(...)`, `schema()`, `payload_indexes()`, `train_codec()`, `health(recall_sample=100)`, `upkeep()`, `compact()` and `rebuild()` (HnswIndex), `train_lists()` (FlatIndex) | See the [developer reference](developer-reference.md#4-hnswindex-the-graph-index). `health()` returns a dict. |
| `save(path)`, `HnswIndex.load(path)`, `HnswIndex.open_durable(dir, dim, metric, options=DurableOptions())`, `checkpoint()` | Files and durability. |
| `len(index)`, `id in index`, `index.dim`, `index.metric` | |

**Distances** follow the library's convention: smaller is closer. For L2 they are **squared** distances, for inner product minus the dot product, for cosine 1 − cosine similarity.

## Coming from hnswlib

Code written for [hnswlib](https://github.com/nmslib/hnswlib) runs on hnsw_lite by changing one import:

```python
from hnsw_lite_lib import hnswlib   # instead of: import hnswlib

p = hnswlib.Index(space="cosine", dim=384)
p.init_index(max_elements=100_000, ef_construction=200, M=16)
p.add_items(vectors, ids)
p.set_ef(50)
labels, distances = p.knn_query(queries, k=10)
```

`Index` and `BFIndex` have hnswlib's methods, arguments, return types and shapes, errors (`RuntimeError`) and distance conventions (squared L2; `1 - dot` for "ip"; `1 - cosine` for "cosine"): `init_index`, `add_items` (an existing label is replaced), `knn_query` (with `filter` and `num_threads`), `set_ef` and `ef`, `mark_deleted` and `unmark_deleted`, `resize_index`, `get_items`, `get_ids_list`, `get_current_count`, `get_max_elements`, `save_index` and `load_index`; `BFIndex` adds `delete_vector`. Running the same code against hnswlib 0.8.0 (the differential tests in `test_hnswlib_compat.py`): `BFIndex` labels identical in every space and distances within 10^-6; `Index` the same neighbors (graphs differ, as between hnswlib builds), distances equal where the rows agree; and the same counts, item access, deletions and errors.

What differs, on purpose:

- **Files**: `save_index` writes hnsw_lite's format in one file; hnswlib's files cannot be loaded, nor ours by hnswlib.
- **`BFIndex.knn_query` with more neighbors than stored vectors** raises `RuntimeError`, as `Index` does; hnswlib's `BFIndex` reads past its results there (undefined behavior).
- **Not provided**: pickling, `index_file_size`, and hnswlib's graph internals.

Kept on purpose, so switching changes no numbers: hnswlib's `BFIndex` with "cosine" normalizes stored vectors but not the query, so its distances are `1 - |q| cos`; the ranking is cosine's. This module returns the same. Under the hood every hnsw_lite feature remains available through `HnswIndex` and `Collection`.

## Arrow, Parquet and importers

**Arrow and Parquet** (needs `pyarrow`). A collection exports to Arrow in the columns `to_pandas()` uses: `id`, `key` and `expires_at` when used, `vector` as a fixed-size list of float32, `sparse` as a struct of `indices` and `values`, `text`, then one column per metadata field (renamed `metadata_<name>` on a clash), with one schema derived from the collection's field types:

```python
table = col.to_arrow()                       # a pyarrow.Table
for batch in col.arrow_batches(batch=10_000):  # RecordBatches: bounded memory
    ...
col.write_parquet("docs.parquet")            # streamed, batch by batch
copy.upsert_parquet("docs.parquet")          # read back, batch by batch
copy.upsert_arrow(table)                     # or from any Arrow table
```

`upsert_arrow` takes the standard column names by default and any others by argument (`id_column=`, `vector_column=`, ...); vectors may be fixed-size or variable lists, nulls are missing values, and a row without a vector is stored without one. Values convert directly rather than through pandas, so an integer column with missing values stays integers (pandas would make it floats, which an integer field refuses). A round trip reproduces a collection exactly: every side, metadata types, and expiry times.

**Importers** (`hnsw_lite_lib.importers`) copy existing data into a new index or collection, which builds its own graph:

| Function | Reads | Needs |
|---|---|---|
| `from_hnswlib(path, space)` → `HnswIndex` | An hnswlib index file, directly: labels, vectors, `M` and `ef_construction`; elements marked deleted are left out. `space` is the one the file was built with (hnswlib does not store it). `read_hnswlib(path)` returns the raw labels, vectors and settings. | Nothing |
| `from_faiss(index_or_path, metric=None)` → `HnswIndex` | A FAISS index or file: its vectors, and its IDs through `IndexIDMap`; IVF indexes are reconstructed through a direct map. Indexes storing compressed codes give their reconstructions. Pass `metric=Metric.COSINE` for vectors normalized for cosine. | `faiss-cpu` |
| `from_chroma(collection, batch=1000)` → `Collection` | A Chroma collection, page by page: IDs as string keys, embeddings, documents as text, metadata; the metric from its `hnsw:space`. | A Chroma client |

Checked against the libraries themselves: vectors read from hnswlib's own files equal its `get_items`, and hnswlib loads files written in the layout the reader expects; four FAISS index types give exactly FAISS's exact neighbors after import. A damaged or hostile hnswlib file (truncated, a flipped header byte, a header claiming terabytes) is refused with `ValueError` before anything is allocated.

## Filters

```python
F = hl.Filter
f = F.eq("lang", "en") & (F.ge("year", 2020) | F.has_tag("tags", "featured")) & ~F.eq("draft", True)
```

| Function | Matches |
|---|---|
| `Filter.eq`, `ne`, `lt`, `le`, `gt`, `ge(field, value)` | comparisons; `value` is an int, float, bool or str (NumPy scalars too) |
| `Filter.between(field, low, high)` | low ≤ field ≤ high (numbers) |
| `Filter.in_(field, values)` | any of the values (strings, integers, numbers or booleans) |
| `Filter.has_tag(field, tag)`, `has_any_tag(field, tags)`, `has_all_tags(field, tags)` | tag sets |
| `Filter.exists(field)`, `Filter.all()` | has a value; everything |
| `Filter.geo_radius(lat_field, lon_field, lat, lon, meters)` | within `meters` of a point, by great-circle distance (latitude and longitude in two numeric fields, degrees) |
| `Filter.geo_box(lat_field, lon_field, south, west, north, east)` | inside a box, edges included; west > east crosses the antimeridian |
| `a & b`, `a \| b`, `~a` | and, or, not (Python's `and`, `or` and `not` cannot be overloaded: use the operators) |

Anywhere a filter is accepted, a **dict** works too: `{"lang": "en", "year": [2023, 2024]}` means `lang == "en"` and `year` in (2023, 2024). `hl.as_filter(f)` does that conversion explicitly (a `Filter` or `None` is returned as it is), for building on a dict filter with `&`, `|` and `~`. The [filter rules](developer-reference.md#34-filter) apply: for example, a comparison never matches a record that lacks the field.

## How Python values are converted

| Python | Library |
|---|---|
| Vectors: NumPy arrays of any float dtype, any memory layout, lists | `float32`, copied (non-contiguous or Fortran-ordered arrays are fine) |
| IDs: `int`, NumPy integers, arrays of integers | unsigned 64-bit (the full range up to 2⁶⁴−1); negative IDs raise `ValueError` (never wrapped around), non-integers `TypeError` |
| Metadata `bool` | `Bool` (checked before `int`, since `bool` is a subclass of `int` in Python) |
| `int`, NumPy integers | `Int` (64-bit) |
| `float`, NumPy floats | `Float` (NaN and infinity are rejected) |
| `str` | `Keyword` |
| `list`, `tuple`, `set` of `str` | `Tags` |
| `None` in `set_metadata` | clears the field |
| Sparse vectors: `{term: weight}` or `(indices, values)` | `SparseVector` |
| Search results | NumPy arrays (indexes) or lists of dicts (collections) |

## Enumerations and option classes

`hl.Metric` (`L2`, `INNER_PRODUCT`, `COSINE`), `hl.Codec` (`NONE`, `SQ8`, `BINARY`, `PQ`, `F16`, `BF16`, `RABITQ`, for `QuantizationParams.codec`; `F16` and `BF16` with `keep_full_vectors = False` halve vector memory), `hl.Strategy` (`AUTO`, `EXACT`, `GRAPH`, `ACORN`), `hl.Preset`, `hl.Fusion`, `hl.Durability`, `hl.Maintenance` (`MANUAL`, `BACKGROUND`) and `hl.FieldType` mirror the C++ enumerations. The option classes `hl.HnswParams`, `hl.QuantizationParams`, `hl.IvfParams`, `hl.DurableOptions`, `hl.MaintenancePolicy` and `hl.CollectionOptions` have the C++ fields as attributes (durations in milliseconds: `interval_ms`, `budget_ms`, `group_interval_ms`; `checkpoint_age_s` in seconds):

```python
p = hl.HnswParams()
p.M = 32
p.quantization.codec = hl.Codec.SQ8
index = hl.HnswIndex(dim=384, metric=hl.Metric.COSINE, params=p)
```

`hl.CollectionOptions` (for `hl.Collection(options=...)`) also has `bm25` (its BM25 settings, reached as `options.bm25`; the class is not exported as `hl.Bm25Params`), with `stem`, `stop_words` and `unicode` (as the arguments of the same names) and `k1` (1.2: how quickly repeated words stop adding to a score), `b` (0.75: how much long texts are discounted) and `avg_doc_length` (100 tokens); and `set_schema(fields, strict=False)`, which takes the same `{"name": type}` dict as the constructor's `schema` (with `strict=True`, fields outside it are refused).

## Threads, errors and limits

**Threads.** Every search, batch add, save, checkpoint and maintenance round releases the GIL, so Python threads run them in parallel. Conversions from Python objects (metadata dicts, lists) happen before the GIL is released. Batch methods with a `threads` argument also parallelize inside C++.

**Errors.**

| Python exception | When |
|---|---|
| `ValueError` | Invalid input: wrong dimension, NaN, a duplicate ID, metadata not fitting the schema, invalid settings, an unknown preset or metric name, using a closed collection |
| `TypeError` | A value of the wrong Python type (a metadata value that is not int, float, bool, str or a list of str) |
| `hl.FormatError` (a `RuntimeError`) | A damaged or foreign file or directory |
| `hl.Cancelled` (a `RuntimeError`) | A long operation stopped by a deadline |
| `RuntimeError` | Files that cannot be opened or written, a directory locked by another process, misuse such as a change from inside a search |
| `MemoryError` | Out of memory (the change is rolled back) |

**Limits.** Everything in the [developer reference's limits](developer-reference.md#11-limits-at-a-glance) applies. In Python specifically:
- Search predicates (Python functions deciding per result) are not exposed: filter with metadata instead. Python callbacks would have to reacquire the GIL for every candidate.
- Vectors are copied into the library (no zero-copy views of its memory).

## Releasing

The `Wheels` workflow (`.github/workflows/wheels.yml`) builds and tests every wheel on every pull request that touches the library or the package, and publishes to PyPI when a version tag is pushed.

**Once, before the first release.** On [pypi.org](https://pypi.org), under *Your account > Publishing*, add a **pending trusted publisher**: project name `hnsw-lite-lib`, the GitHub owner and repository, workflow `wheels.yml`, environment `pypi`. Then, in the GitHub repository's *Settings > Environments*, create an environment named `pypi` (optionally requiring a reviewer's approval, so every release is confirmed by a person). No password or token is stored anywhere: PyPI trusts this workflow directly.

**Each release.**

1. Change `__version__` in `python/hnsw_lite_lib/__init__.py` (the only place the version is written: `pyproject.toml`, `CMakeLists.txt` and `conanfile.py` read it), set the same version in `ports/hnsw-lite/vcpkg.json`, and merge.
2. Tag the merged commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

3. The workflow builds the 20 wheels (CPython 3.9 to 3.13 on four platforms) and the source distribution, tests each wheel in a clean environment, checks that the tag matches `__version__` (refusing to publish otherwise), checks the files with `twine`, and uploads them.
4. In `ports/hnsw-lite/portfile.cmake`, set `SHA512` to the new tag's archive: with `SHA512 0`, `vcpkg install hnsw-lite --overlay-ports=ports` stops and prints the right value.

PyPI never accepts the same version twice: a mistake is fixed by releasing the next version.