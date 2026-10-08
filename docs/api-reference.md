# API reference

Every public class and function, grouped by header. All names are in the `vecdb` namespace. For how to use them together, see the [User guide](user-guide.md).

Most programs need only `FlatIndex` or `HnswIndex` with `SearchOptions`, `Filter` and `Metadata`. The lower layers (`Storage`, `VectorStore`, `get_distance()` and the rest) are public too, for building other kinds of index.

All code is in the `vecdb` namespace. Invalid input throws `std::invalid_argument`, `std::out_of_range` or `std::length_error`.

## `common.h`

| Name | Description |
|---|---|
| `NodeId` | Internal vector number (`std::uint32_t`) |
| `kEmpty` | Largest `NodeId`; marks an empty neighbor slot |
| `kAlign` | Alignment in bytes (64) |
| `kFloatsPerLine` | Floats per cache line (16) |
| `round_up(n, m)` | Rounds `n` up to a multiple of `m` |

## `AlignedBlock`

| Member | Description |
|---|---|
| `AlignedBlock(bytes, fill = 0)` | Allocates `bytes` (rounded up to 64), 64-byte aligned, filled with `fill` |
| `data()` | Start address |
| `size()` | Size in bytes |

Not copyable, movable. Frees its memory in the destructor.

## `Arena`

| Member | Description |
|---|---|
| `Arena(block_bytes = 1 MB)` | Creates an empty arena |
| `allocate(bytes, align)` | Returns `bytes` of memory aligned to `align` (power of two, at most 64) |
| `allocate_array<T>(n)` | Returns memory for `n` objects of plain type `T` |
| `block_count()` | Number of blocks allocated so far |

## `VectorStore`

| Member | Description |
|---|---|
| `VectorStore(dim, shelf_bits = auto)` | Creates a store for vectors of `dim` floats; blocks default to at most 65,536 rows and 8 MB |
| `add(span<const float>)` | Copies a vector in, returns its `NodeId` |
| `get(id)` | Read-only `span` of `dim` floats (no copy) |
| `get_padded(id)` | Read-only `span` of `stride` floats, including padding |
| `size()`, `dim()`, `stride()` | Number of vectors, dimension, padded row length |
| `rows_per_shelf()` | Rows per memory block |
| `undo_last_add()`, `pop_back()` | Removes the last row; its memory is reused by the next `add` |
| `overwrite(id, vector)`, `move_row(from, to)` | Replaces a row (slot reuse); copies a row (swap-with-last) |

## `IdMap`

| Member | Description |
|---|---|
| `add(external)` | Registers a user ID, returns the next `NodeId`; throws on duplicates |
| `find(external)` | `std::optional<NodeId>` for a user ID |
| `external(id)` | User ID of a `NodeId` |
| `mark_deleted(id)`, `is_deleted(id)` | Sets or checks the deletion flag |
| `undo_last_add(external)` | Undoes the most recent `add`; used to roll back a failed insert |
| `release(external)` | Real deletion: frees the user ID and marks its slot free; returns the slot |
| `is_free(slot)`, `bind(slot, external)` | Checks a slot is free; gives a free slot a new user ID (slot reuse) |
| `move_slot(from, to)`, `pop_back_slot()` | Used by Flat's swap-with-last removal |
| `size()` | Number of registered IDs |

## `GraphStorage`

| Member | Description |
|---|---|
| `GraphStorage(M = 16, shelf_bits = 16)` | Creates storage with `M` links per upper level and `2M` on level 0 |
| `add_node(level)` | Adds the next node with the given top level (0 to 255), returns its `NodeId` |
| `links(id, level)` | Writable `span` of the node's slots on that level; throws if the node does not reach it |
| `count(slots)` | Number of used slots (static) |
| `set_links(id, level, neighbors)` | Replaces the node's list on that level and marks the remaining slots empty |
| `undo_last_add()` | Undoes the most recent `add_node`, before any links were written |
| `reset_node(id, level)` | Clears a node and gives it a new level (slot reuse); all-or-nothing |
| `level(id)` | Top level of a node |
| `size()`, `M()`, `M0()` | Number of nodes and list capacities |

## `Storage`

| Member | Description |
|---|---|
| `Storage(dim, M = 16)` | Creates all Layer 1 parts |
| `insert(external, vector, level)` | Validates the dimension and level (0 to 255), then adds the vector to `IdMap`, `VectorStore` and `GraphStorage`; returns its `NodeId`. On error nothing is added |
| `insert_into(slot, external, vector, level)` | Puts a vector into a free slot instead of appending; all-or-nothing |
| `vectors()`, `graph()`, `ids()` | Access to the parts |
| `size()` | Number of stored vectors |

## `distance.h`

| Name | Description |
|---|---|
| `enum class Metric { L2, InnerProduct, Cosine }` | Which distance to compute |
| `enum class Isa { Scalar, Avx2, Avx512, Neon }` | Which instruction set a kernel uses |
| `DistanceFn` | `float (*)(const float* a, const float* b, std::size_t n)` |
| `get_distance(metric)` | Fastest kernel for this CPU; never `nullptr` |
| `get_distance(metric, isa)` | A specific version, or `nullptr` if this build or CPU cannot run it |
| `isa_supported(isa)` | True if `isa` is compiled in and this CPU can run it |
| `active_isa()` | The version `get_distance(metric)` uses |
| `isa_name(isa)` | `"scalar"`, `"avx2"`, `"avx512"` or `"neon"` |
| `normalize(span<float>)` | Scales a vector to length 1 in place, computing in double so very small vectors work; a zero vector is left unchanged |

## `search_result.h`

| Name | Description |
|---|---|
| `SearchResult { id, distance }` | One result for the user: their 64-bit ID and the distance |
| `Candidate { distance, id }` | Internal result with a `NodeId`; ordered by distance, then id |
| `TopK(k)` | Keeps the k closest candidates (reserves memory for at most 1024 up front, so any k is safe) |
| `TopK::push(c)` | Adds `c` if there is room or it beats the worst kept; returns whether it was kept |
| `TopK::full()`, `size()`, `worst_distance()` | Heap state; `worst_distance()` is infinity when empty |
| `TopK::take_sorted()` | Empties the heap and returns candidates closest first |
| `ordered_distance(d)` | Returns +∞ for NaN, `d` otherwise, so overflowed distances sort last |
| `CompactStats { kept, reclaimed }` | What `compact()` did |

## `PreparedVector`

| Member | Description |
|---|---|
| `PreparedVector(dim)` | Creates a zeroed, 64-byte-aligned buffer of `stride` floats |
| `prepare(vector, metric)` | Copies `vector` in and normalizes it for cosine; throws on a wrong dimension, NaN or infinity, leaving the previous contents unchanged |
| `data()` | Start of the padded row, for a `DistanceFn` |
| `values()`, `padded()` | Spans of `dim` or `stride` floats |
| `dim()`, `stride()` | Sizes |

## `VisitedList` and `VisitedListPool`

| Member | Description |
|---|---|
| `VisitedList::reset(node_count)` | Starts a new search; grows the array if needed |
| `VisitedList::visit(id)` | Marks `id`; returns true on the first visit in this search |
| `VisitedList::visited(id)` | Whether `id` was visited in this search |
| `VisitedListPool::acquire(node_count)` | Borrows a reset list inside a `Handle` that returns it automatically |
| `VisitedListPool::idle_count()` | Lists currently waiting in the pool |

## `FlatIndex`

| Member | Description |
|---|---|
| `FlatIndex(dim, metric)` | Creates an empty exact index |
| `add(id, vector)` | Stores a vector; throws on a wrong dimension, NaN or infinity, or a used ID |
| `remove(id)` | Deletes it for real (swap-with-last); the ID can be added again; never allocates |
| `compact()`, `deleted_count()`, `capacity()` | No-op (never any holes), 0, equal to `size()` |
| `contains(id)` | Stored and not removed |
| `search(query, k)` | Up to k exact closest live vectors, closest first; k above `size()` returns all |
| `size()`, `dim()`, `metric()` | Live count and settings |

## `HnswIndex`

| Member | Description |
|---|---|
| `HnswParams { M = 16, ef_construction = 200, seed = 42, repair_on_remove = true }` | Graph settings; `M >= 2` and `ef_construction >= 1` are required, `M >= 8` is recommended |
| `HnswIndex(dim, metric, params = {})` | Creates an empty index; throws on invalid params |
| `add(id, vector)` | Inserts into the graph; throws on a wrong dimension, NaN or infinity, or a used ID |
| `remove(id)` | Repairs the graph around it (if enabled), frees the ID for reuse and its slot for the next insert |
| `compact()` | Rebuilds a dense index from the live vectors; the old index stays intact if memory runs out |
| `deleted_count()`, `capacity()` | Slots waiting to be reused; all slots |
| `contains(id)` | Stored and not removed |
| `search(query, k, ef = 64)` | Up to k approximate closest live vectors; `ef` is raised to at least k and capped at the number of nodes |
| `size()`, `dim()`, `metric()`, `params()` | Live count and settings |
| `max_level()`, `entry_point()`, `storage()` | Graph inspection, mainly for tests |

## `metadata.h`

| Name | Description |
|---|---|
| `FieldType` | `Int`, `Float`, `Bool`, `Keyword`, `Tags` |
| `Value::from(x)` | Builds a value; the type comes from the C++ type of `x` |
| `Metadata().set(name, value)`, `set_tags(name, tags)`, `unset(name)` | Builds the fields of one vector; `unset` clears a field in `set_metadata` |
| `Schema::strict(fields)`, `Schema::dynamic(fields = {})` | Schema modes; empty or duplicate names throw |
| `MetadataStore` | Column storage: `validate`, `write`, `update`, `clear`, `move_row`, `read`, `index_field`, `value_count`, `present_count`, `compact_dictionary`, `unused_string_count`, fast column accessors |

## `filter.h`

| Name | Description |
|---|---|
| `Filter::eq/ne/lt/le/gt/ge(field, value)`, `between`, `in`, `has_tag`, `has_any_tag`, `has_all_tags`, `exists`, `all()` | Conditions; combine with `&&`, `\|\|`, `!` |
| `CompiledFilter(filter, store)` | Checks a filter against a schema; `matches(slot)`, `exact_count()` |
| `SearchOptions { filter, predicate, strategy, planner, stats, range_expand_outside }` | Optional settings for one search |
| `Strategy { Auto, ForceExact, ForceGraph }` | How a filtered search runs |
| `PlannerParams { max_exact_matches = 2000, max_exact_fraction = 0.01, sample_size = 256, max_ef_multiplier = 32 }` | Planner settings |
| `SearchStats` | Strategy, reason, selectivity, estimated matches, `ef`, nodes visited, distances computed |

## New index methods (`FlatIndex` and `HnswIndex`)

| Member | Description |
|---|---|
| `FlatIndex(dim, metric, schema)`, `HnswIndex(dim, metric, params, schema)` | The schema is optional (dynamic by default) |
| `add(id, vector, metadata = {})` | Stores a vector with optional metadata |
| `set_metadata(id, metadata)`, `get_metadata(id)` | Changes given fields only / reads a vector's metadata |
| `create_payload_index(field)` | Exact value counts for a keyword, boolean or tag-set field |
| `search(query, k, options)` (Flat), `search(query, k, ef, options)` (HNSW) | Filtered search |
| `search_batch(queries, k, threads, options)` (Flat), `search_batch(queries, k, ef, threads, options)` (HNSW) | Batch search |
| `search_range(query, radius, max_results = kNoLimit, options)` | Range search |
| `set_planner_params`, `planner_params` (HNSW) | Index-wide planner settings |

`thread_pool.h` provides `ThreadPool` (`parallel_for`, exceptions rethrown to the caller), `SharedPool` and `effective_threads` (the thread cap); `Filter::kMaxDepth` is the nesting limit; `GraphStorage::upper_slots_allocated()` and `recycled_upper_blocks()` and `HnswIndex::pooled_visited_lists()` expose memory diagnostics; `search_result.h` adds `kNoLimit` and `l2_radius`.