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
| `add_unmapped(external)`, `occupy_unmapped(slot, external)` | A live slot holding a user ID without being findable (the new version during an update) |
| `repoint(external, slot)`, `retire(slot)` | Switches a user ID to another slot; frees a slot `find()` no longer points to |
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
| `insert_unmapped(...)`, `insert_into_unmapped(slot, ...)` | The same, without making the user ID findable (used by update); all-or-nothing |
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
| `add_batch(ids, vectors, [metadata], threads)` | Adds many vectors, all or nothing; on the calling thread |
| `search_by_id(id, k, ...)`, `remove_batch(ids)` | Search with a stored record's vector (empty optional if not stored; the record itself left out unless `include_self`); remove many IDs, skipping unknown ones. Do not range-for over `*search_by_id(...)` directly: store the optional first |
| `get_vector(id)`, `get(id)`, `scan(options)` | Read back one vector (exactly as given; cosine indexes keep the original), one record, or pages of records in ascending ID order (`ScanOptions`: `filter`, `limit`, `after`, `with_vectors`, `with_metadata`; `Page`: `records`, `next`) |
| `remove_where(filter)`, `set_metadata_where(filter, changes)`, `count_where(filter)` | Remove, change metadata of, or count every vector matching a filter; return how many matched. `remove_where` always completes; `set_metadata_where` is all or nothing |
| `remove(id)` | Deletes it for real (swap-with-last); the ID can be added again; never allocates |
| `update(id, vector)`, `update(id, vector, metadata)` | Replaces the vector in place (and, with metadata, the whole metadata); `false` for an unknown ID; all-or-nothing |
| `upsert(id, vector)`, `upsert(id, vector, metadata)` | Adds or replaces; returns `true` if added |
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
| `add_batch(ids, vectors, [metadata], threads)` | Adds many vectors, all or nothing, linking them on `threads` threads (0: all cores); one thread gives the same index as `add` in order |
| `search_by_id(id, k, ...)`, `remove_batch(ids)` | Search with a stored record's vector (empty optional if not stored; the record itself left out unless `include_self`); remove many IDs, skipping unknown ones. Do not range-for over `*search_by_id(...)` directly: store the optional first |
| `get_vector(id)`, `get(id)`, `scan(options)` | Read back one vector (exactly as given; cosine indexes keep the original), one record, or pages of records in ascending ID order (`ScanOptions`: `filter`, `limit`, `after`, `with_vectors`, `with_metadata`; `Page`: `records`, `next`) |
| `remove_where(filter)`, `set_metadata_where(filter, changes)`, `count_where(filter)` | Remove, change metadata of, or count every vector matching a filter; return how many matched. `remove_where` always completes; `set_metadata_where` is all or nothing |
| `remove(id)` | Repairs the graph around it (if enabled), frees the ID for reuse and its slot for the next insert |
| `update(id, vector)`, `update(id, vector, metadata)` | Builds and links the new version in another slot, then switches the ID over and frees the old slot; `false` for an unknown ID; a failure leaves the old vector intact |
| `upsert(id, vector)`, `upsert(id, vector, metadata)` | Adds or replaces; returns `true` if added |
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
| `Value::of_int`, `of_float`, `of_bool`, `of_keyword`, `of_tags` | A metadata value of one exact type (`Value::from` picks the type from a C++ value) |
| `Metadata().set(name, value)`, `set_tags(name, tags)`, `unset(name)` | Builds the fields of one vector; `unset` clears a field in `set_metadata` |
| `Schema::strict(fields)`, `Schema::dynamic(fields = {})` | Schema modes; empty or duplicate names throw |
| `MetadataStore` | Column storage: `validate`, `write`, `update`, `clear`, `move_row`, `read`, `index_field`, `value_count`, `present_count`, `compact_dictionary`, `unused_string_count`, fast column accessors |

## `filter.h`

| Name | Description |
|---|---|
| `Filter::eq/ne/lt/le/gt/ge(field, value)`, `between`, `in`, `has_tag`, `has_any_tag`, `has_all_tags`, `exists`, `all()` | Conditions; combine with `&&`, `\|\|`, `!` |
| `filter.is_all()`, `filter.root()` | Whether a filter matches everything (`Filter::all()`), and its condition tree, for inspection |
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
| `set_planner_params`, `planner_params` (HNSW) | Index-wide planner settings; `planner_params()` returns a copy, safe while other threads change them |

`filter.h` also holds two helpers the indexes use internally: `plan_search` (the planner's choice between graph and exact search) and `validate_batch` (checks a batch of queries). `thread_pool.h` provides `ThreadPool` (`parallel_for`, exceptions rethrown to the caller), `SharedPool` and `effective_threads` (the thread cap); `Filter::kMaxDepth` is the nesting limit; `GraphStorage::upper_slots_allocated()` and `recycled_upper_blocks()` and `HnswIndex::pooled_visited_lists()` expose memory diagnostics; `search_result.h` adds `kNoLimit` and `l2_radius`.

## Hardening additions

| Name | Header | Description |
|---|---|---|
| `IndexLock` | `thread_pool.h` | The lock inside every index: reads in parallel, changes one at a time, writers preferred. `ReadGuard`, `WriteGuard` (throws `std::logic_error` inside any index operation), `Hold` (batch tasks), `held()`, `inside_any()`, `kMaxNesting` (64) |
| `index_distance(raw, cosine)` | `search_result.h` | A distance as indexes report it: cosine clamped at 0, NaN mapped like `ordered_distance()`. `get_distance()` kernels are unchanged |
| `in_pool_task()` | `thread_pool.h` | True on a thread running a pool task; `parallel_for()` called there runs inline, so nested batches cannot deadlock |
| `HnswParams::M` | `hnsw_index.h` | Now limited to 2 to 2048 |

## Saving and loading

| Name | Description |
|---|---|
| `index.save(path)` | Writes the whole index to `path`, through a temporary file of its own (`<path>.tmp-` and a unique suffix) and a rename, so an existing file is never damaged and simultaneous saves to one path each produce a complete file. Throws `std::runtime_error` if the file cannot be written |
| `index.save(std::ostream&)` | Writes to a stream |
| `HnswIndex::load(path)`, `FlatIndex::load(path)` | Return a `std::unique_ptr` to the loaded index. Throw `FormatError` for a damaged, truncated, unsupported or wrong-kind file, and `std::runtime_error` if the file cannot be opened |
| `HnswIndex::load(std::istream&)`, `FlatIndex::load(std::istream&)` | The same from a seekable stream |
| `FormatError` (`serialize.h`) | Derived from `std::runtime_error` |
| `MetadataStore::save(w)`, `MetadataStore::load(r, slots, live)` | Used by the indexes to save and load metadata |
| `persist::Writer`, `persist::Reader`, `persist::crc32`, `persist::StreamErrorsThrow` (`serialize.h`) | The format's building blocks; see [Architecture and design](architecture.md#saving-and-loading) |

## Deadlines, cancellation and progress

| Name | Description |
|---|---|
| `SearchOptions::deadline`, `SearchOptions::cancel` | Stop a search at a moment, or when an `std::atomic<bool>` becomes true; it returns the best results found so far |
| `SearchStats::stopped_early` | Whether a deadline or cancel flag stopped the search |
| `OperationOptions` (`deadline`, `cancel`, `progress`) | Control of `add_batch`, `compact` and `save`; `progress(done, total)` is called in order, at most about 100 times |
| `Cancelled` (a `std::runtime_error`) | Thrown by a cancelled or late operation, after undoing it |

## Training and codecs

These are the building blocks of compressed indexes (see the [compression and index types plan](compression-and-index-types-plan.md)).

| Name | Description |
|---|---|
| `kmeans(data, dim, KMeansParams)` | Clusters `data` (n * dim values) into `k` centroids; returns `KMeansResult` (`centroids`, `assignment`, `inertia`, `iterations_run`). Deterministic for a given seed, on every CPU and thread count |
| `KMeansParams` | `k`, `iterations` (default 25), `seed`, `threads` (0: all cores) |
| `nearest_centroid(centroids, dim, point, distance)` | Index of the nearest centroid (squared L2; ties to the lower index) |
| `VectorCodec` | Interface: `kind()`, `code_size()`, `trained()`, `train(sample)`, `encode`, `decode`, `prepare(query, table)`, `distance(table, code)` |
| `SQ8Codec` | 8-bit scalar quantization: one byte per dimension; `minimum()`, `step()`, `set_range()` |
| `BinaryCodec` | Binary quantization: one bit per dimension, with a learned threshold and spread per dimension |
| `PQCodec(dim, metric, parts = 0)` | Product quantization: one byte per part, 256 centroids per part learned by k-means; `parts()`, `centroids()` |
| `CodecKind`, `make_codec(kind, dim, metric, pq_subspaces = 0)` | Codec kinds (`None`, `SQ8`, `Binary`, `PQ`) and a factory |
| `get_code_kernels()`, `get_code_kernels(isa)`, `CodeKernels` | The SIMD kernels behind code distances (`sq8_l2`, `sq8_dot`, `bits_sum`): the fastest for this CPU, or one instruction set (for tests and benchmarks) |
| `QuantizationParams` | `codec` (`None`, `SQ8`, `Binary`, `PQ`), `train_after` (default 1,000; 0: only explicit), `rerank` (default on), `rerank_factor` (default 4), `truncate_to` (Matryoshka truncation; 0: off), `keep_full_vectors` (default on; off drops the copy of each vector as given), `pq_subspaces` (PQ parts; 0: one per 8 dimensions); in `HnswParams::quantization`, and the last `FlatIndex` constructor argument |
| `train_codec()`, `codec_trained()`, `get_code(id)`, `codec()`, `has_vector_values()` | On both indexes: train now (all or nothing; refused once only codes are kept), whether searches use codes, one stored code, the codec itself (for inspection), and whether full-precision vectors are still kept |
| `VectorCodec::state()`, `restore(state)`, `clone()` | The trained state as numbers (saved in files), restoring it, and a copy |
| `dim()`, `indexed_dim()` | On both indexes: the dimension of vectors and queries as given, and the dimensions stored and searched (fewer with truncation) |
| `indexed_dim(params, dim)`, `prepare_truncated(out, v, input_dim, metric)` | The dimensions an index with these settings stores (throws if the truncation exceeds `dim`); preparing a full input vector for a truncating index |

## Inverted file (IVF)

| Name | Description |
|---|---|
| `IvfParams` | `lists` (0: none), `probes` (default 8), `train_after` (default 10,000; 0: only explicit); the last `FlatIndex` constructor argument |
| `train_lists()`, `lists_trained()` | Train the lists now (k-means; all or nothing; may be repeated), and whether searches use them |
| `list_of(id)`, `list_sizes()`, `ivf()` | The list holding a vector, the size of each list, and the settings |
| `SearchOptions::probes` | Lists to probe for one search (0: the index's default) |

## Disk-based index

| Name | Description |
|---|---|
| `DiskIndex::build(path, ids, vectors, dim, metric, params, metadata, schema)` | Builds a Vamana graph and PQ codes, and writes the index file (replacing `path` only once complete). L2 and cosine |
| `DiskIndex::build_from_file(path, vectors_path, metric, LargeBuildParams, ids, metadata, schema, &stats)` | Builds from a `.fbin` vectors file within a memory budget (partitions merged when it does not fit) |
| `LargeBuildParams`, `LargeBuildStats` | `index`, `memory_budget`, `replicas` (at least 2 with several partitions), `partitions` (0: automatic); `partitions`, `largest_partition`, `merged_and_pruned` |
| `write_vectors_file(path, vectors, dim)` | Writes vectors as a `.fbin` file |
| `DiskIndex::open(path, DiskOpenOptions)` | Opens an index file, verifying everything held in memory |
| `DiskOpenOptions` | `io_threads` (read each round's nodes concurrently; 0: one after another), `simulated_latency` (delays every read, for tests and benchmarks) |
| `search(query, k, DiskSearchParams{L, beam_width}, options, &stats)` | Searches from disk; filters, predicates, deadlines; exact distances |
| `search_batch`, `search_by_id`, `contains`, `get_vector`, `get`, `scan` | As for the in-memory indexes |
| `read_node(node)`, `blocks_read()`, `medoid()`, `codec()`, `metadata()` | Inspection: one node as stored, blocks read since opening, the entry point, the PQ codec, the metadata |
| `DiskIndexParams` | `graph` (`VamanaParams`), `pq_subspaces` (0: one part per 2 dimensions) |
| `DiskSearchStats` | `nodes_read`, `blocks_read`, `rounds`, `stopped_early` |
| `VamanaGraph::build(vectors, dim, metric, VamanaParams{R, L, alpha, seed})` | The Vamana graph alone, built and searched in memory (`search`, `neighbors`, `medoid`) |
| `BlockFile` | Positional reads from a read-only file, safe from many threads |

## Health reports

| Name | Description |
|---|---|
| `health(HealthOptions)` | On `HnswIndex`, `FlatIndex` and `DiskIndex`: a `HealthReport`; shared lock (searches continue), then the sampled searches |
| `HealthOptions` | `recall_sample` (100; 0: no estimate), `k` (10), `ef` (64), `seed`, `codec_sample` (500: vectors for compression and IVF figures), `retrain_sample` (0: no retraining comparison) |
| `HealthReport` storage | `live`, `slots`, `free_slots` |
| `HealthReport` graph | `levels` (`nodes`, `expected` per level), `mean_degree`, `min_degree`, `underfull`, `no_incoming`, `unreachable`, `links_to_removed`, `stale_links` |
| `HealthReport` drift | `codec_error`, `codec_error_retrained`, `clipped` (SQ8); `lists`, `lists_trained`, `largest_list`, `empty_lists`, `list_imbalance`, `centroid_distance`, `centroid_distance_retrained` |
| `HealthReport` quality and churn | `recall` (optional), `removals`, `updates`, `churn()` |

## Upkeep

| Name | Description |
|---|---|
| `upkeep(UpkeepOptions, OperationOptions)` | On `HnswIndex`: consolidates links to removed vectors and reconnects unreachable ones; returns `UpkeepStats`. A deadline or cancellation stops it early without error |
| `UpkeepOptions` | `consolidate` (true), `reconnect` (true), `max_nodes` (0: no limit; vectors repaired per call) |
| `UpkeepStats` | `nodes_repaired`, `links_replaced`, `reconnected`, `unreachable` (afterwards), `complete` (then the churn counters are reset) |
| `rebuild(RebuildOptions, OperationOptions)` | On `HnswIndex` and `FlatIndex`: retrains the codec (and IVF lists) while the index stays in use, then swaps atomically; returns `RebuildStats`. Cancelled, or out of memory: unchanged. One at a time (`std::logic_error`) |
| `RebuildOptions`, `RebuildStats` | `retrain` (true); `kept`, `reclaimed`, `caught_up` (IDs changed during the rebuild), `codec_retrained`, `lists_retrained` |
| `set_auto_compact(fraction)`, `auto_compact()` | Automatic `compact()` once free slots exceed `fraction` of all slots (0: off; at most 0.9); not saved |

## Durability

| Name | Description |
|---|---|
| `open_durable(dir, ...)` | On `HnswIndex` (`dir, dim, metric, params, schema, options, stats`) and `FlatIndex` (`dir, dim, metric, schema, quantization, ivf, options, stats`): creates or recovers a durable directory |
| `checkpoint()`, `durable()` | A snapshot and a new log segment (returns its LSN); whether the index is durable |
| `DurableOptions` | `keep_snapshots` (2), `durability` (`Buffered`, `Grouped`, `Synced`), `group_interval` (10 ms) |
| `RecoveryStats` | `created`, `snapshot_lsn`, `fallbacks`, `segments`, `replayed`, `torn_bytes`, `dropped_unfinished` |
| `attach_log(WalWriter)`, `detach_log()`, `replay(records)` | Lower level: log a non-durable index's changes yourself; replay returns whether an unfinished final operation was dropped |
| `WalWriter`, `wal::read_segment`, `WalScan`, `WalRecord` | Log segments: appending, syncing, reading with torn-tail detection |

## Sparse vectors and hybrid search

| Name | Description |
|---|---|
| `SparseVector` | `indices` (terms), `values` (weights); `sparse_dot(a, b)` |
| `SparseIndex(schema)` | `add`, `remove`, `remove_batch`, `remove_where`, `update`, `upsert`, `set_metadata`, `contains`, `size`, `get_vector`, `get_metadata`, `records`, `search(query, k, options)`, `document_frequency(term)`, `terms`, `postings`, `dead_postings`, `slots`, `compact`, `save`, `load` |
| `Bm25(Bm25Params)` | `k1` (1.2), `b` (0.75), `avg_doc_length` (100); `fit(documents)`, `encode_document(text)`, `encode_query(text, index)`, `tokenize`, `term_id` |
| `hybrid_search(dense, dense_query, sparse, sparse_query, k, options)` | Dense and sparse searches fused by ID; the dense index is any `HnswIndex` or `FlatIndex` |
| `fuse(dense_results, sparse_results, k, options)` | Fuses two ranked lists |
| `HybridOptions` | `fusion` (`Rrf`, `Weighted`), `rrf_k` (60), `dense_weight`, `sparse_weight` (1), `dense_k`, `sparse_k` (0: max(4k, 20)), `dense_ef` (0: max(dense_k, 64)), `filter`, `dense_options`, `sparse_options` |
| `HybridResult` | `id`, `score` (higher is better), `dense_rank`, `sparse_rank` (1-based), `dense_distance`, `sparse_score` |

## Supporting types

| Name | Description |
|---|---|
| `CompareOp` | `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, for comparison filters |
| `FieldSpec`, `SchemaMode` | A field's name and type in a `Schema`; `Strict` (only declared fields) or `Dynamic` (fields created on first use) |
| `PlanDecision`, `PlanReason` | What the query planner chose (strategy, estimated matches and selectivity, beam width) and why, from `plan_search()` |
| `LevelHealth` | Nodes on one HNSW level against the number expected, in `HealthReport` |
| `SparseRecord` | `id`, `vector`, `metadata`, from `SparseIndex::records()` |
| `Fusion` | `Rrf` or `Weighted`, in `HybridOptions` |
| `DiskIndex::max_degree()`, `id_of(node)`, `node_of(id)` | The graph's maximum degree, and conversions between internal node numbers and user IDs |

## Collections (`collection.h`)

| Name | Description |
|---|---|
| `Collection(options)`, `Collection::open(dir, options, stats)` | In memory, or a durable directory (created or recovered) |
| `add`, `upsert`, `remove`, `remove_batch`, `remove_where`, `set_metadata`, `set_metadata_where` | Changes, each all or nothing; `upsert` replaces the whole document |
| `contains`, `size`, `count_where`, `get(id)` | Reading; `get` returns a `DocumentRecord` (`vector`, `text`, `sparse`, `metadata`) |
| `search(query, k, options)` | Any combination of `CollectionQuery::vector`, `text` and `sparse`; returns `CollectionHit`s |
| `checkpoint()`, `durable()`, `save`, `load`, `replay` | Durability and files |
| `dense_index()`, `text_index()`, `sparse_index()` | Read-only access to the sides (null if absent) |
| `CollectionOptions` | `dim` (0: no dense side), `metric` (cosine), `hnsw`, `schema`, `text` (true), `bm25`, `sparse` (false), `store_text` (true), `durability` |
| `Document` | `vector`, `text`, `sparse`, `metadata`, each optional |
| `CollectionSearchOptions` | `filter`, `predicate`, `fusion`, `rrf_k`, `vector_weight`, `text_weight`, `sparse_weight`, `candidates` (0: max(4k, 20)), `ef`, `with_metadata`, `with_text` |
| `CollectionHit` | `id`, `score`, `vector_rank`, `text_rank`, `sparse_rank`, `vector_distance`, `text_score`, `sparse_score`, `metadata`, `text` |

