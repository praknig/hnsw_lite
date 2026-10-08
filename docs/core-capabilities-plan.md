# Core capabilities: design plan

This document plans four capabilities that hnsw-lite lacks compared with production vector databases:

1. Real deletion
2. Updating vectors
3. Saving and loading
4. Concurrent inserts

The test scenarios for these designs are in [core-capabilities-test-plan.md](core-capabilities-test-plan.md).

Status: **1. Real deletion is implemented**; 2 to 4 are planned.

---

## Implementation order

The features depend on each other, so the order matters.

| Order | Feature | Why here |
|---|---|---|
| 1 | Real deletion | Changes the core data structures (free slots, reusable IDs), so it must happen before anything is saved to disk |
| 2 | Updating vectors | Built directly on top of deletion |
| 3 | Saving and loading | Captures the final shape of the data structures, so the file format is designed once instead of changing right after release |
| 4 | Concurrent inserts | Comes last because it must protect every operation, including those added in steps 1 to 3 |

---

## 1. Real deletion

### Today

`remove()` only sets a flag. The vector keeps its memory, its graph node stays, and its ID can never be used again. Under heavy churn (constant adds and removes), memory grows forever and search slowly degrades as the graph fills with dead nodes.

### Goals

- A removed ID can be added again.
- New inserts reuse the slots of removed vectors, so memory is bounded by the peak number of live vectors.
- The graph is repaired so dead nodes stop degrading search.
- An explicit `compact()` rebuilds a fully dense index on request.

### FlatIndex: swap-with-last removal

Flat has no graph, so removal can be immediate: move the last row into the removed row's slot, update both IDs in the map, and shrink the count by one. No tombstones, constant time, and memory is reused automatically.

Side effect: internal order no longer matches insertion order after removals, so equal distances are ordered by internal position rather than insertion order. This is documented behavior.

### HnswIndex: free list, graph repair and slot reuse

Swap-with-last is not possible for the graph, because other nodes' links point to specific slots.

1. **On remove:** release the user ID immediately so it can be reused, keep the slot as a tombstone, and push the slot onto a **free list**.
2. **Repair the neighbors** (on by default, can be turned off): for each neighbor of the removed node, re-select that neighbor's links from its current links plus the removed node's links, minus the removed node. This is hnswlib's approach. Because links are mostly two-way, it removes most links pointing to the dead node and bridges the gap it leaves.
3. **On insert:** if the free list is not empty, reuse a slot (last in, first out). Overwrite the vector, draw a new random level, reset the node's links and link it normally. If the new level is higher than the slot's old level, the node gets a new arena block; the old block is abandoned, which the arena design already allows.
4. **Entry point:** removing the entry point selects a new one, a live node on the highest live level, and lowers the top level if needed. Finding it scans the level array, but this happens only when the entry point itself is removed.

**Decision — removing every vector:** when no live vector remains, the entry point becomes "none" (`kEmpty`) and the top level becomes −1, exactly as in a new index. Dead slots stay in storage and on the free list; the next insert reuses one and becomes the entry point.

**Why not track reverse links?** Storing which nodes link *to* each node would make removal exact, but roughly doubles graph memory and complicates every insert. Neighbor repair is what production libraries use; its small remaining imperfection is measured by the churn tests.

### `compact()`

Rebuilds the index from live vectors only, with fresh dense internal numbers. For HNSW this is a full rebuild: slow but predictable, and an explicit maintenance step. The new index is built completely before replacing the old one, so a failure leaves the old index intact. Returns statistics (vectors kept, memory freed). For Flat it is a no-op, since swap-with-last never leaves holes.

### Changes per layer

- **Layer 1:** `IdMap` gains "release a user ID but keep the slot" and "bind a new user ID to an existing slot". `VectorStore` gains "overwrite a row", "move a row" and "shrink by one". `GraphStorage` gains "reset a node with a new level". A free-list structure is added.
- **Layer 3:** the remove, insert and entry-point logic above, `compact()`, and statistics: `deleted_count()` and `capacity()`.

---

## 2. Updating vectors

### Today

Impossible: a vector must be removed and added under a different ID.

### API

- `update(id, vector)` replaces the vector for an existing ID. **Decision:** it returns `false` for an unknown or removed ID, matching how `remove()` behaves.
- `upsert(id, vector)` adds the vector if the ID is new and replaces it otherwise.

Two methods make the intent explicit at the call site.

### FlatIndex

Validate and prepare the new vector, then overwrite the row in place. All-or-nothing, because nothing can fail after validation.

### HnswIndex

- **Remove, then re-insert (chosen).** Built entirely on real deletion. Because the free list is last-in-first-out, the re-insert reuses the same slot.
- **In-place update with local repair** (hnswlib's approach). Faster for small moves but much more complex; deferred until benchmarks show updates are a bottleneck.

**Safety:** an update must never lose the old vector. All risky work (validation, finding the new neighbors, reserving memory) happens before the old node is touched, so a failure leaves the old vector intact.

---

## 3. Saving and loading

### Today

Everything is lost when the program exits, and rebuilding a large HNSW index takes minutes.

### API

`save(path)` and a static `load(path)` that returns a new index, plus stream versions (`save(std::ostream&)`, `load(std::istream&)`) that make testing possible without files.

### File format (version 1)

A versioned binary format:

| Section | Contents |
|---|---|
| Header | Magic string, format version, byte order, index type, dimension, metric, HNSW parameters |
| IDs | User IDs, deletion flags, free list |
| Vectors | Real values only; padding is recreated on load |
| Graph | Each node's level and link lists |
| Index state | Entry point, top level, live count, and the random generator's state |
| Checksums | One per section |

Saving the random generator's state means that adding vectors after loading builds exactly the same graph as if the program had never stopped, preserving the "same seed, same graph" guarantee across a save.

**Decision — extra bytes after the data:** files with anything after the last section are **rejected**. Accepting them could hide corruption or a truncated-then-appended file.

### Raw pointers in the graph

`GraphStorage` keeps raw pointers into the arena for upper-level links, and pointers cannot be written to a file meaningfully.

- **Version 1 (chosen):** save the graph *logically* (each node's level and lists) and rebuild the arena on load. Simple, portable, independent of memory layout.
- **Later optimization:** memory-mapping the file for instant loading would require the arena to store offsets instead of pointers. Only worth doing if load time becomes a problem.

### Files are untrusted input

A damaged or malicious file must never cause a crash or an out-of-bounds access:

- Every size is checked against the actual file size **before** allocating, so a corrupted count cannot trigger a huge allocation.
- Every link, level, count, flag and free-list entry is validated after reading.
- Any problem throws a clear format error. A failed load never produces a partly built index.

`save` writes to a temporary file and renames it into place at the end, so a crash during saving never destroys the previous good file.

### Compatibility

- Byte order is recorded in the header; files with a different byte order are rejected.
- A graph saved on one CPU loads on another: kernels round slightly differently, but the graph remains valid.
- Small version 1 "golden files" are committed to the repository, so CI verifies that future versions can still read them.

---

## 4. Concurrent inserts

### Today

One writer at a time, and no searches while writing. Building is single-threaded (50,000 vectors take 8.6 seconds).

This is the largest and riskiest item, so it is split into two stages.

### Stage A: one lock around the whole index

A shared mutex: searches take it in shared mode, so many run at once; adds, removes, updates, saves and compaction take it exclusively. Writes still happen one at a time, but searches and writes can be mixed freely from any number of threads. Internal operations call unlocked versions of each other (for example, `update` calls the unlocked remove and add) so the lock is never taken twice.

### Stage B: fine-grained locking (parallel inserts)

Following hnswlib:

- **One lock per node's link list**, using a fixed array of spinlocks shared by many nodes (lock striping) to keep memory small.
- **One lock for the entry point and top level.**
- **An atomic counter** to hand out new slots.
- **Memory ordering:** a node's vector and empty lists are fully written before any link to it is published, so a search can never reach a half-built node.

Layer 1 changes this requires:

- `VectorStore` and `GraphStorage` keep their shelves in a `std::vector`, which can reallocate while another thread reads it. They need a **fixed-size shelf table** that never moves. Filling the table is a clean error.
- `IdMap`'s hash map gets its own lock; its arrays move to the same stable chunked storage, because searches read deletion flags and user IDs while inserts grow them.
- Deletion flags become atomic bytes.
- The arena gets a lock, or one arena per thread.

**Deletion under concurrency:** reusing a slot while other threads may be traversing it is very hard to get right. In stage B, slot reuse, repair and compaction run in exclusive mode, while plain adds and searches run in parallel.

### `add_batch`

`add_batch(ids, vectors, threads)` builds in parallel. **Decision:** it is **all-or-nothing**. Every vector is validated and every ID checked (against the index and within the batch) before anything is inserted; if any check fails, nothing is inserted.

### Determinism

With several threads, insertion order varies between runs, so concurrently built graphs differ from run to run. Single-threaded builds, including `add_batch` with one thread, remain fully reproducible.

---

## Cross-cutting

- **New API:** `update`, `upsert`, `compact`, `save`, `load`, `add_batch`, `deleted_count`, `capacity`.
- **Exception safety:** every new operation is either all-or-nothing or documents exactly what state it leaves, and is covered by out-of-memory sweeps.
- **Tests:** new groups `deletion`, `update` and `persistence` in the comprehensive suite; the `concurrency` group grows. Each group gets its own CTest entry and runs in CI on every platform; the ThreadSanitizer job covers `concurrency`, `deletion` and `update`.
- **Documentation:** each feature adds usage to the [User guide](user-guide.md), a design section to [Architecture and design](architecture.md), entries to the [API reference](api-reference.md), and removes the matching items from the limitations in [Project notes](project-notes.md#limitations).

## Branches and effort

| Branch | Estimated effort |
|---|---|
| `feature/real-deletion` | 3 to 4 days |
| `feature/vector-update` | 1 to 2 days |
| `feature/persistence` | 3 to 4 days |
| `feature/concurrent-writes` | Stage A: half a day; stage B: 4 to 6 days |

Each feature gets its own pull request, so CI checks it in isolation.

## Risks

- **Stage B concurrency** is where subtle bugs hide. ThreadSanitizer in CI and long stress tests are the main defense; stage A is a safe fallback if stage B takes longer than expected.
- **Graph repair quality** might degrade recall slowly over many removals in ways short tests miss. A long churn test with a recall threshold guards against this.
- **File format mistakes are permanent** once files exist. The format must be reviewed carefully before merging; golden-file tests keep it honest afterwards.

## Decisions

| Question | Decision |
|---|---|
| Entry point after removing every vector | "None" (`kEmpty`) and top level −1, as in a new index; dead slots stay on the free list |
| Extra bytes after the data in a saved file | Reject the file |
| An invalid item inside `add_batch` | All-or-nothing: insert nothing |
| `update` on an unknown or removed ID | Return `false`, matching `remove` |