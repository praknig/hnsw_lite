# Core capabilities: test plan

Test scenarios for the four features designed in [core-capabilities-plan.md](core-capabilities-plan.md): real deletion, updating vectors, saving and loading, and concurrent inserts. The last section analyzes how these features affect the existing tests.

Status: **section 1 (real deletion, D1 to D66) is implemented** in the `deletion` group, with every test named after its scenario ID; sections 2 to 4 are planned. Each scenario has an ID so it can be traced to its test once written. All tests go into `tests/test_comprehensive.cpp`.

**Types:** **P** positive (normal use) · **N** negative (invalid input or misuse) · **E** edge case · **S** stress · **M** out-of-memory sweep

---

## 1. Real deletion — new group `deletion`

### Layer 1 building blocks

| ID | Scenario | Type |
|---|---|---|
| D1 | `IdMap::release` frees the user ID (lookup fails) but keeps the slot | P |
| D2 | Releasing an unknown or already released ID reports failure and changes nothing | N |
| D3 | Binding a new user ID to a free slot: lookups work in both directions | P |
| D4 | Binding a user ID that is already in use is rejected; nothing changes | N |
| D5 | Binding to a slot that is not free is rejected | N |
| D6 | The free list returns slots last-in-first-out; an empty free list means "append" | P/E |
| D7 | `VectorStore` overwrite: new values stored, padding still zero, address unchanged | P |
| D8 | Overwrite with a wrong dimension or unknown slot is rejected; the row is unchanged | N |
| D9 | `VectorStore` move row: destination equals source, padding still zero | P |
| D10 | `VectorStore` shrink, then add: the row is reused and no new shelf is allocated | P |
| D11 | `GraphStorage` reset at the same level: every list becomes empty | P |
| D12 | Reset at a higher level: new upper lists exist and are empty; lower lists are cleared | P |
| D13 | Reset at a lower level: levels above the new top are no longer accessible | E |
| D14 | Reset with level −1 or 256 is rejected; the node is unchanged | N |
| D15 | Out-of-memory sweeps for bind, reset and overwrite: all-or-nothing | M |

### FlatIndex: swap-with-last removal

| ID | Scenario | Type |
|---|---|---|
| D20 | Remove from the middle: results match the reference (same IDs and distances) | P |
| D21 | Remove the last vector: no row is moved | E |
| D22 | Remove the only vector, then add again | E |
| D23 | Remove everything in random order, re-add everything: same results as a fresh index | P |
| D24 | Remove A (which moves B into A's slot), then remove B: both handled correctly | E |
| D25 | A removed ID can be added again immediately, with a different vector | P |
| D26 | `remove` returns false for unknown and already removed IDs; `contains` is false after removal | N |
| D27 | The stored count always equals the live count (Flat has no tombstones) | P |
| D28 | 10,000 random adds and removes checked against a simple `std::map` reference after every step | S |
| D29 | IDs 0 and the maximum 64-bit value can be removed and reused | E |
| D30 | Out-of-memory sweep of remove: either fully removed or fully unchanged | M |

### HnswIndex: free list, repair and slot reuse

| ID | Scenario | Type |
|---|---|---|
| D40 | A removed ID can be added again; the new vector is found, the old one never | P |
| D41 | Inserting after a removal reuses the slot: the stored count does not grow | P |
| D42 | A reused slot gets a higher level than before: the new upper lists are valid | E |
| D43 | A reused slot gets a lower level than before | E |
| D44 | After repair, the removed node's former neighbors no longer link to it, and their lists are valid | P |
| D45 | After removing 30% of vectors, at least 99% of live vectors remain reachable | P |
| D46 | With repair turned off, behavior matches tombstones and recall stays acceptable | P |
| D47 | Removing the entry point: the new entry point is live and on the highest live level | E |
| D48 | Removing every vector: the entry point is `kEmpty` and the top level −1 (as in a new index), search returns nothing, dead slots stay in storage, and the next insert reuses a slot and becomes the entry point | E |
| D49 | Remove everything, re-add everything under the same IDs: recall comparable to a fresh build | P |
| D50 | **Churn:** 20 cycles of removing and re-adding 20%; recall, graph validity and memory checked every cycle | S |
| D51 | Remove a whole cluster, then search that area: results come from what remains | E |
| D52 | Churn for inner product and cosine, with metric-appropriate thresholds | S |
| D53 | Identical vectors: removing some copies leaves the others reachable | E |
| D54 | Removal with `M = 2` still leaves a structurally valid graph | E |
| D55 | The same sequence of operations with the same seed gives identical results | P |
| D56 | Out-of-memory sweep of remove (repair allocates): index consistent, ID state consistent | M |
| D57 | Out-of-memory sweep of an insert into a reused slot | M |

### `compact()` and statistics

| ID | Scenario | Type |
|---|---|---|
| D60 | HNSW compact: every live vector kept, user IDs unchanged, recall against Flat still high | P |
| D61 | After compact: the stored count equals the live count and `deleted_count()` is 0 | P |
| D62 | Compact on an empty index, one with no deletions, and one with everything deleted | E |
| D63 | Compact returns correct statistics (vectors kept, memory freed) | P |
| D64 | Out-of-memory sweep of compact: the old index is left fully intact | M |
| D65 | Flat compact is a no-op and reports nothing freed | E |
| D66 | `size()`, `deleted_count()` and `capacity()` stay consistent through every operation | P |

---

## 2. Updating vectors — new group `update`

`update` returns `false` for an unknown or removed ID, matching `remove`.

| ID | Scenario | Type |
|---|---|---|
| U1 | Flat update: searches find the new vector, never the old position | P |
| U2 | Flat update of an unknown or removed ID returns `false`; nothing changes | N |
| U3 | Flat update with NaN, infinity or a wrong dimension throws `std::invalid_argument`; the old vector is intact | N |
| U4 | Flat update to the identical vector: results unchanged | E |
| U5 | Flat update with cosine: the new vector is normalized | P |
| U6 | Flat upsert: a new ID is added, an existing ID is replaced; sizes correct | P |
| U7 | HNSW update: the new position is found, the old position never returned | P |
| U8 | HNSW update keeps the stored count, and grows capacity by at most one slot, which later updates reuse (changed from "reuses the same slot": see the plan's "As built" note) | P |
| U9 | HNSW update of an unknown or removed ID returns `false`; nothing changes | N |
| U10 | HNSW update with invalid input: the old vector is intact, entry point and top level unchanged | N |
| U11 | Updating the entry point's vector: the entry point stays valid and live | E |
| U12 | Updating every vector once: recall against Flat high, graph valid | S |
| U13 | Updating the same ID 1,000 times: memory stable, graph valid | S |
| U14 | Small moves versus large moves (nearby versus far across the space) | E |
| U15 | Updating to a vector identical to another stored one | E |
| U16 | Updating in an index containing a single vector | E |
| U17 | HNSW upsert semantics, matching Flat | P |
| U18 | All three metrics | P |
| U19 | The same operations and seed give identical results | P |
| U20 | **Out-of-memory sweep of update: the old vector survives every failure** | M |

Added during implementation (metadata did not exist when this plan was written): U21 update keeps metadata, U22 update with metadata replaces every field, U23 upsert with and without metadata, U24 invalid metadata changes nothing, U25 the new ID-map and storage operations, U26 updates without graph repair, U27 5,000 mixed operations against a reference.

---

## 3. Saving and loading — new group `persistence`

### Round trips

| ID | Scenario | Type |
|---|---|---|
| P1 | Flat, every metric: results after loading identical to before saving | P |
| P2 | HNSW, every metric: identical results, entry point, top level and every link | P |
| P3 | Round trip after removals, after updates and after compaction | P |
| P4 | Empty index, both types | E |
| P5 | Single vector | E |
| P6 | Dimensions 1, 37 (not a multiple of 16) and 1,536 | E |
| P7 | Extreme user IDs survive the round trip | E |
| P8 | Settings preserved: dimension, metric, `M`, `ef_construction`, seed | P |
| P9 | **Continuation:** add 500, save, load, add 500 gives the same graph as adding 1,000 without saving | P |
| P10 | The free list survives: after loading, the next insert reuses the same slot | P |
| P11 | Saving to a stream and to a file produce identical bytes | P |
| P12 | Saving the same index twice produces identical bytes | P |
| P13 | Saving over an existing file replaces it; no temporary files are left behind | P |

### Invalid and damaged files

Every case must produce a clean format error: never a crash, never a partly built index.

| ID | Scenario | Type |
|---|---|---|
| P20 | **Truncate a saved file at every byte position** | N |
| P21 | Flip bits throughout the file: the checksum catches it | N |
| P22 | Wrong magic string, unsupported future version, wrong byte order | N |
| P23 | A Flat file loaded as HNSW, and the reverse | N |
| P24 | Empty file; file containing only a header | N |
| P25 | Any bytes after the last section: the file is rejected | N |
| P26 | Inflated counts (for example, claiming a billion vectors): rejected **before** allocating anything | N |
| P27 | Invalid content with valid checksums, each rejected: links out of range, self-links, levels above 255, invalid entry point, invalid deletion flags, invalid free list (duplicates, live slots), duplicate user IDs | N |
| P28 | Saving to an invalid path: clean error | N |
| P29 | Out-of-memory sweep of load: clean error, no partial index | M |
| P30 | Out-of-memory sweep of save: clean error, **destination file unchanged** | M |

### Compatibility

| ID | Scenario | Type |
|---|---|---|
| P40 | Version 1 golden files committed to the repository load correctly and give known results | P |
| P41 | A file saved on one platform loads on another (CI: Linux saves; Windows and macOS load) | P |
| P42 | Large index (100,000 vectors) round trip completes and matches | S |

---

## 4. Concurrent inserts — expanded group `concurrency`

### Stage A: one lock around the index

| ID | Scenario | Type |
|---|---|---|
| C1 | Searches while one thread adds: no crash; every result sorted, finite, and an ID that was added | S |
| C2 | Several threads adding different ID ranges: final count correct, all present, graph valid, recall high | S |
| C3 | Mixed adds, removes and searches from many threads: final sizes match the operations applied | S |
| C4 | Two threads add the same ID: exactly one succeeds, the other is rejected | N |
| C5 | Two threads remove the same ID: exactly one returns true | N |
| C6 | Update during searches: a result never contains the same ID twice | S |
| C7 | Save during searches: the file loads and is consistent | P |
| C8 | Compact during searches: searches afterward are correct | P |
| C9 | No deadlock when operations call each other internally (update calling remove and add), enforced by a watchdog timeout | N |
| C10 | The same scenarios for Flat | S |

### Stage B: fine-grained locking

`add_batch` is all-or-nothing: if any vector or ID in the batch is invalid, nothing is inserted.

| ID | Scenario | Type |
|---|---|---|
| C20 | `add_batch` with 1, 2, 4 and 8 threads: all present, graph valid, recall high | P |
| C21 | `add_batch` with one thread gives exactly the same graph as sequential adds | P |
| C22 | A duplicate ID inside the batch, or one already in the index: nothing is inserted and the index is unchanged | N |
| C23 | An invalid vector (NaN, infinity, wrong dimension) anywhere in the batch: nothing is inserted | N |
| C24 | 8 threads adding while 8 search, for a fixed duration: every invariant holds afterward | S |
| C25 | Growth across many shelves (tiny shelf size) under concurrency: addresses stable, no torn reads | S |
| C26 | Many threads inserting high-level nodes: the entry point is valid and on the top level | E |
| C27 | Deletion flags read by searches while removes happen | S |
| C28 | Concurrent user ID lookups and inserts in `IdMap` | S |
| C29 | The fixed shelf table is full: clean error, no corruption | E |
| C30 | Single-threaded builds stay fully reproducible | P |

**CI:** the ThreadSanitizer job runs the `concurrency`, `deletion` and `update` groups, so data races are caught on every pull request.

---

## 5. Impact on existing tests

Real deletion changes some behavior on purpose, which affects 7 existing tests and 1 shared helper. Updating vectors and saving and loading only add new APIs and change no existing behavior.

### Must change: the behavior changes on purpose

| Existing test | Why | Required change |
|---|---|---|
| `flat.removed_id_not_reusable` | Removed IDs become reusable | Invert into `flat.removed_id_reusable` |
| `hnsw.removed_id_not_reusable` | Same | Invert into `hnsw.removed_id_reusable` |
| `hnsw.new_nodes_do_not_link_to_removed` | Assumes new vectors get internal numbers above 300; with slot reuse they take the freed numbers, so the test would check the wrong nodes | Rewrite: find new vectors through their user IDs, and check the stronger property that no live node links to a removed one after repair |
| `check_graph` (helper used by about 25 tests) | Requires every node's level to be at most the top level; after the entry point is removed, the top level can drop below the level of a dead node still in storage | Skip that check for removed nodes |

### Likely to need small adjustments

| Existing test | Why | Change |
|---|---|---|
| `flat.removed_never_returned` | Its reference breaks distance ties by insertion order, but swap-with-last removal reorders internal positions; it passes today only because exact ties are rare with random data | Compare tie-tolerantly: same distances, same set of IDs |
| `hnsw.remove_semantics` | Asserts the stored count is still 2 after a removal; that remains true until a later insert reuses the slot, but the reasoning changes | Update the comment; optionally assert reuse afterward |
| `hnsw.remove_all_then_add` | The state after removing everything is now defined | Assert entry point `kEmpty` and top level −1 after removing everything (scenario D48) |
| `robustness.oom_hnsw_add_keeps_index_consistent` | Uses a fresh ID per attempt because a failed insert could keep its ID taken; if a failure now releases the ID, this can be tightened | Optionally retry with the same ID |

### Unaffected

| Group | Tests | Why |
|---|---|---|
| `layer1` | 56 | New methods are added; existing behavior is unchanged. Stage B rewrites the shelf storage, and these tests become its main regression net (stable addresses, block sizes, padding) |
| `layer2` | 31 | Kernels and dispatch do not change |
| `helpers` | 31 | Unchanged |
| `flat` | 22 of 24 | Only the two listed above |
| `hnsw` | 40 of 44 | Only the four listed above |
| `robustness` | 19 of 20 | The sweeps are generic, so they cover new allocation points automatically |
| `concurrency` | 4 | Locks only make more things safe |
| `e2e` | 6 | Recall should improve with repair and reuse |

### README changes

The Limitations section changes when these features land: "IDs cannot be reused", "No persistence", "No memory reclamation", "One writer at a time", and "A failed HNSW insert can use up its ID" are removed or rewritten.

---

## Summary

| Group | New scenarios |
|---|---|
| `deletion` | 51 |
| `update` | 20 |
| `persistence` | 27 |
| `concurrency` | 21 |
| **Total** | **119** |

Plus changes to 7 existing tests and 1 shared helper.

A later coverage audit added 8 more deletion tests beyond this plan (and 2 Layer 1 tests), bringing the library to 100% line coverage; see [Code coverage](testing.md#code-coverage).