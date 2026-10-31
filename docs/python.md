# How it works

This page explains, in plain terms, how hnsw-lite finds similar vectors quickly: what problem it solves, how its main index is built and searched, what each setting changes, and how filtering, deletion, compression and durability fit in. It assumes no background in search algorithms. For every method and parameter in detail, see the [developer reference](developer-reference.md); for the internals (memory layout, kernels, file formats), see [architecture and design](architecture.md).

## The problem: finding the nearest vectors

Embedding models turn text, images or audio into **vectors**: lists of a few hundred to a few thousand numbers, arranged so that similar things get nearby vectors. Search then means: given a query vector, find the `k` stored vectors closest to it. "Closest" is measured by a **metric**: Euclidean distance (`L2`), the inner product (`InnerProduct`), or the angle between vectors (`Cosine`).

**Exact search** compares the query with every stored vector. hnsw-lite does this in `FlatIndex`, with SIMD kernels that process 8 or 16 numbers at a time. It always returns the true nearest neighbors, but its cost grows with the number of vectors times their dimension: fine for thousands of vectors, too slow for millions when answers are needed in milliseconds.

**Approximate search** looks at a small, well-chosen fraction of the vectors instead, and accepts that it may occasionally miss one of the true nearest neighbors. How often it finds them is measured by **recall@k**: the fraction of the true `k` nearest neighbors in the results. At 1,000,000 vectors of 128 dimensions, hnsw-lite's graph index reaches a recall@10 of 0.998 while answering 4,462 queries per second on one core (median 0.19 ms per query); see [benchmarks](benchmarks.md).

## The graph index (HNSW)

`HnswIndex` implements **HNSW** (Hierarchical Navigable Small World graphs, Malkov and Yashunin, 2016). Each stored vector is a node, linked to a handful of nodes near it. Searching means walking the links: from wherever you are, move to the linked node closest to the query, and repeat. Because every node links to its neighbors, the walk keeps getting closer.

A single flat graph has a weakness: from a distant starting point, a walk takes many small steps. HNSW fixes this with **layers**, like the express and local lines of a transit network:

```
layer 2   o-----------------------o                  few nodes, long links
layer 1   o--------o------o-------o--------o
layer 0   o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o    every node, short links
```

Every node is on layer 0. Each node is also given a random **top layer**, with each layer up holding about `M` times fewer nodes: with `M` 16, about 1 node in 16 reaches layer 1, 1 in 256 reaches layer 2, and so on. (The top layer is `floor(-ln(u) / ln(M))` for a random `u` between 0 and 1.) The upper layers, sparse and with long links, cross the space quickly; layer 0, dense and with short links, finds the precise neighbors.

### Building the graph

Adding a vector:

1. **Find its neighborhood.** Starting from the top of the graph, walk down layer by layer toward the new vector, as a search would.
2. **Search more widely where it will live.** On each layer up to the new node's top layer, keep a list of the best `ef_construction` candidates (200 by default) rather than just the single closest node.
3. **Choose its links.** From those candidates, keep up to `M` links (16 by default; `2 × M` on layer 0, where every node lives). The choice is not simply "the closest `M`": a candidate is skipped if one already chosen is closer to it than the new node is, since the candidate can then be reached through that one. This keeps links pointing in different directions, which is what keeps the graph easy to navigate.
4. **Link back.** Each chosen neighbor links to the new node too; if that overfills its list, its list is re-chosen by the same rule.

Identical vectors get special care: with the plain rule, copies of one vector ended up nearly unreachable (591 of 600 in a test), so copies are exempt from the rule and share the list with distinct neighbors.

### Searching the graph

1. **Descend.** Start at the entry point on the top layer. On each layer above 0, walk greedily: move to the closest linked node until none is closer, then step down a layer.
2. **Search layer 0 with a beam.** Keep the best `ef` candidates found so far (64 by default, and never fewer than `k`), repeatedly expand the most promising one, and stop when no candidate can improve the list.
3. **Return the best `k`.**

The beam width `ef` is the main dial between speed and recall. Measured at 1,000,000 × 128: recall@10 is 0.862 at `ef` 32, 0.970 at 64, 0.998 at 128 and 1.000 at 256. You can choose it per query.

### What the settings change

| Setting | Default | Higher means | Cost |
|---|---|---|---|
| `M` | 16 | Better recall, especially in high dimensions | More memory (each link takes 4 bytes) and slower building |
| `ef_construction` | 200 | A better graph, so better recall at the same `ef` | Slower building only; no effect on memory or search speed |
| `ef` (per search) | 64 | Better recall | Slower searches |

For collections, presets pick these together: `CollectionOptions::preset(dim, Preset::Balanced)` (the default: `M` 16, `ef_construction` 200, search beam 128), `Preset::HighRecall` (32, 400, 256) or `Preset::LowMemory` (12, 150, 128, with vectors stored as SQ8 codes). From a million vectors on, `Balanced` and `HighRecall` raise `M` and `ef_construction` further. See the [user guide](user-guide.md) and [performance and tuning](performance.md).

## Searching with a filter

Searches often come with conditions: only documents in English, only products in stock. Checking the filter on each result after a normal search fails when few vectors match: the `k` closest vectors may all be filtered out. Checking it during the walk has the opposite problem: when matches are rare, the walk crosses many non-matching nodes for each useful one, and can get stuck in a region with none.

So hnsw-lite has a **query planner**. It first estimates how many vectors match (testing every one if there are at most 256, otherwise a fixed sample of 256, so the same query always plans the same way), then chooses:

| Matching vectors | Strategy | Why |
|---|---|---|
| At most 2,000, or at most 1% | **Exact scan** of the matching vectors | Few enough to compare them all: exact and fast |
| Fewer than 20% | **ACORN** | Scores only matching nodes and looks through non-matching neighbors to theirs (two hops), so the walk stays connected; beam 4 times wider |
| 20% or more | **Graph search** that skips non-matching results | Beam widened by 1 / the matching fraction (at most 32 times), since the walk meets that many nodes per useful one |

In measurements on 20,000 clustered vectors, ACORN needed 1.4 to 24 times fewer distance computations than the widened graph search below 20% of matches, with recall from 0.984 to 1.000. Each search can report what the planner chose and why (`SearchStats`), and you can force a strategy.

## Removing and updating vectors

**Removing** a vector hides it from every search at once. With `repair_on_remove` (on by default), its neighbors are relinked to each other, so the graph stays navigable around the gap. The memory is reclaimed by `compact()`, which you can run yourself or leave to automatic maintenance.

**Updating** a vector builds the new version in a fresh place, then switches to it in one step: a search sees the old version or the new one, never a half-changed one. Every change is all or nothing, even when memory runs out.

## Memory, and making it smaller

Each vector costs its numbers (4 bytes each), its links (about `8 × M` bytes) and some bookkeeping: about 4 × dimension + 8 × `M` + 45 bytes, plus metadata. Measured with `M` 16: 687 bytes per vector at 128 dimensions and 3,280 at 768. Ways to need less:

- **Half precision** (float16 or bfloat16): 2 bytes per number, with recall within 0.005 of float32 in our tests. 47% less memory at 768 dimensions.
- **SQ8**: 1 byte per number. **Binary**: 1 bit per number. **PQ**: 1 byte per part of the vector. Each can keep the full vectors to re-rank the best candidates exactly, or drop them to save the most.
- **Matryoshka truncation**: for embedding models trained for it, index only the first part of each vector.
- **The disk index** (`DiskIndex`, in the style of DiskANN): the graph and full vectors stay on disk; only compact codes, IDs and metadata are in memory.
- **IVF** (for `FlatIndex`): vectors grouped into clusters, and only the clusters nearest the query are scanned.

## Making it fast

- **SIMD kernels.** Distances are computed 8 or 16 numbers at a time, with versions for AVX2, AVX-512 and ARM NEON. The library picks the best one for the CPU it runs on, at startup.
- **Graph reordering.** `reorder()` renumbers the nodes so that neighbors sit near each other in memory, so the processor's cache serves more of each walk: the same results, about 24% faster on clustered data, as real embeddings are.
- **Parallel work.** Batches of searches and inserts are spread over a thread pool. Every index and collection can be used from many threads with no locking of your own: searches run in parallel with each other, and changes run one at a time (a steady stream of searches can never starve a change). On an HNSW index, an insertion does its search for neighbors alongside running searches and holds them up only while it links itself in; a collection still holds its searches for the whole of a change.

## Keeping data safe

A **durable** index or collection writes each change to a log before applying it, and periodically saves a full snapshot (a checkpoint). After a crash, opening the directory loads the latest snapshot and replays the log after it, so every acknowledged change is recovered. You choose when changes reach the disk: `Buffered` (fastest), `Grouped` (several changes per disk flush) or `Synced` (every change flushed before it is acknowledged). A lock keeps a second process from opening the same directory.

## Collections and hybrid search

A `Collection` keeps documents with up to three representations: a dense vector, text, and a sparse vector. Text is searched with **BM25**, the classic keyword-ranking formula; sparse vectors, by their overlap with the query's. A **hybrid** search runs each kind and fuses the rankings, by default with **reciprocal rank fusion** (each list contributes `weight / (60 + rank)` per document, so a document ranked well in several lists rises to the top). Collections also manage metadata, filters, expiry times and string keys.

## Where to go next

- [User guide](user-guide.md): how to use everything described here
- [Performance and tuning](performance.md): choosing settings for your data
- [Benchmarks](benchmarks.md): measuring on your own machine
- [Architecture and design](architecture.md): the internals
- [Python](python.md): the same, from Python