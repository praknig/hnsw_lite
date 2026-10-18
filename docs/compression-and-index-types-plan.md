# Compression and index types plan

The plan for items 4.1 to 4.4 of the [feature catalog](feature-catalog.md): quantization (SQ8, binary and product quantization), shorter vectors (Matryoshka truncation), an IVF index, and a disk-based index in the style of DiskANN.

## Contents

- [Why these belong together](#why-these-belong-together)
- [Phases](#phases)
- [Decisions](#decisions)
- [File format version 3](#file-format-version-3)
- [Testing](#testing)
- [How the work is organized](#how-the-work-is-organized)
- [Risks](#risks)
- [Status](#status)

## Why these belong together

All four share two foundations, built once in phase Q0:

- **A codec layer:** how each vector is stored (full floats, 8-bit codes, bits, or product-quantized codes) and how a distance is computed from that form. Today every vector is stored as full floats; the codec makes that one choice among several.
- **Training:** SQ8 learns each dimension's value range, while PQ and IVF learn centroids with k-means. Training happens once, from sample data, before codes can be computed.

## Phases

| Phase | Contents | Effort |
|---|---|---|
| **Q0. Foundations** | A `VectorCodec` interface; a training API (explicit `train(sample)`, or automatic from the first vectors added); a k-means routine (deterministic for a given seed, parallel); file format version 3; a benchmark reporting recall, memory and build time per setting | 3–4 days |
| **Q1. SQ8** | 8 bits per dimension, with per-dimension ranges learned from data. Distances are computed between the float query and the byte codes (asymmetric distance, more accurate than comparing codes), with SIMD kernels for AVX2, AVX-512 and NEON and a plain C++ fallback. About 4 times less memory for vectors | 3–4 days |
| **Q2. Matryoshka truncation** | Index only the first N dimensions of embeddings trained for it (an option per index); queries are truncated the same way, and cosine vectors re-normalized after truncation | 1 day |
| **Q3. Binary quantization** | 1 bit per dimension (its sign), with Hamming distance computed by `popcount`. About 32 times less memory, so re-ranking at full precision is essential for good results | 2–3 days |
| **Q4. Product quantization (PQ)** | Each vector is split into *m* parts, each coded as one of 256 centroids learned per part; distances use lookup tables built once per query. 16 to 64 times less memory | 1 week |
| **Q5. IVF index** | A second index type: k-means divides the space into cells, each holding a list of its vectors, and a search visits the `nprobe` nearest cells. First with full vectors in the lists, then combined with PQ as IVF-PQ (as in Faiss). It shares metadata, filters, locking, saving, deadlines and the record functions with the existing indexes | 1 week |
| **Q6. Disk-based index** | A Vamana graph (as in DiskANN) whose full vectors and links live on disk in 4 KB blocks, with PQ codes in memory to guide the search and disk reads batched per search step. Read-only in its first version: built, saved, loaded and searched, with changes requiring a rebuild. Updating it in place (as FreshDiskANN does) would be a later project. Full-precision vectors for re-ranking move to disk here (decision D6) | 4–6 weeks |

**Total: roughly 9 to 12 weeks.** The catalog's earlier estimate of "3+ weeks" for the disk-based index was optimistic: the on-disk layout, batched reads on Linux, macOS and Windows, and building indexes larger than memory each take real work.

Phases Q1 to Q4 are independent of each other once Q0 exists; Q5 uses Q0's k-means and, for IVF-PQ, Q4; Q6 uses Q4 (PQ codes guide its search) and Q0's codec layer.

## Decisions

| # | Decision | Options | Status |
|---|---|---|---|
| D6 | Where full-precision vectors live when an index is quantized (for re-ranking and `get_vector`) | (a) in memory as well: exact re-ranking, little memory saved; (b) not kept: the most memory saved, no re-ranking, `get_vector` returns decoded approximations; (c) on disk, read only for re-ranking candidates | **Decided: (a) by default, then (c) with Q6; (b) as a per-index setting**, in two steps: `keep_full_vectors = false` first drops the copy of each vector as given (cosine originals, full vectors of truncating indexes; done), then the full-precision vectors of SQ8 indexes (done) |
| D7 | How HNSW builds its graph when vectors are quantized | From full floats (a better graph) or from codes (less memory while building) | **Decided: from floats** whenever they are available (D6 a or c) |
| D8 | When training happens | An explicit `train(sample)` before adding; automatic, on the first N vectors; or both | **Decided: both**: automatic by default, explicit for control |
| D9 | What `get_vector` returns for a truncated (Matryoshka) index | The full vector as given, or only the indexed dimensions | **Decided: the full vector**, keeping the existing promise that `get_vector` returns a vector exactly as given |
| D10 | IVF's first version | Full vectors in the lists, or IVF-PQ from the start | **Decided: full vectors first**, then IVF-PQ once Q4 exists |

Decisions D3 (Flat batch modes) and D4 (SQ8 as an option per index) from the [hardening and performance plan](hardening-and-performance-plan.md#decisions-needed) still apply; D4 is implemented by Q1.

## File format version 3

Version 2 already added the original vectors of cosine indexes, so these features need **version 3**:

- a section describing the codec and its trained state (SQ8 ranges, PQ codebooks), and the codes themselves;
- the Matryoshka setting (the number of indexed dimensions);
- new index kinds for IVF (kind 3) and the disk-based index (kind 4). The disk-based index is itself a file layout, so it is opened in place rather than read into memory.

Versions 1 and 2 stay readable, and new golden files for version 3 join the frozen ones for versions 1 and 2, as before.

## Testing

Each phase follows the usual pattern: test scenarios written first, edge cases, out-of-memory sweeps, ThreadSanitizer and AddressSanitizer, coverage, and documentation, with every claim checked. Compression adds:

- **Quality thresholds:** recall against exact results, for each codec, metric and setting, with and without re-ranking.
- **Fixed-answer tests** for codes and distances, so that every CPU's SIMD code produces identical codes (needed for the golden files to be identical on every platform).
- **Training edge cases:** too few vectors to train on, all-identical vectors, constant dimensions, values outside the trained range after training (clamped), and training on a filtered or empty sample.
- **For the disk-based index:** files larger than the memory available to the test, damaged and truncated files, and reads failing partway.

One limitation: the development environment used so far can measure recall, memory and correctness reliably, but not speed (it is a shared single core). The speed benefits, especially of the SIMD kernels and the disk-based index, must be measured on a quiet machine with dedicated cores.

## How the work is organized

All phases are developed on one branch, `feature/compression-and-index-types`, with **one commit per phase** (the branch's first commit holds this plan). Every commit is complete on its own: it builds, passes every test, and includes its documentation, so the branch is never broken and each phase can be reviewed by itself (`git show` of one commit).

Because the branch will be long-lived, `master` is merged into it whenever `master` changes, and CI runs on every push.

## Progress notes

**Q0 (done).** `kmeans` (k-means++ seeding, Lloyd's refinement, empty clusters refilled from the farthest point) and the `VectorCodec` interface, with SQ8 as its first implementation in plain C++, so the interface is proven by a real codec. Both are deterministic across CPUs, compilers and thread counts. `kmeans.cpp` and `codec.cpp` are compiled with `-ffp-contract=off` (GCC and Clang): built for ARM64 without it, SQ8 distances differed in their last bits from x86, while with it every result is bit-identical (checked by fingerprinting results on x86, ARM with the flag and ARM without). **Q1** therefore covers SQ8's SIMD kernels, its integration into the indexes (training, encoding on add, distances during search, re-ranking) and file format version 3.

**Q1 (done, except SIMD kernels).** SQ8 in both indexes: `QuantizationParams` (`HnswParams::quantization`, and a new `FlatIndex` constructor argument); automatic training at `train_after` vectors and explicit `train_codec()`, both all or nothing, on a deterministic sample of up to 100,000 vectors; codes following every slot path (adds, reuse, updates, removals including Flat's swap with last, batches and their rollbacks, compaction, which keeps the training); searches comparing codes, then re-ranking at full precision; and file format version 3 (a `CODC` section). Measured recall@10 on 3,000 clustered vectors of 32 dimensions: with re-ranking, SQ8 matched full precision exactly for every metric (HNSW 0.982 to 0.985; Flat 1.000); with codes alone it lost about 1% (HNSW 0.970 to 0.978; Flat 0.987 to 0.993). As decided (D6 a), full-precision vectors stay in memory, so SQ8 adds 25% memory for now; the saving comes with Q6, or sooner with option D6 b. **SIMD kernels (done since).** `sq8_l2`, `sq8_dot` and `bits_sum` in AVX2, AVX-512 and NEON, chosen like the float kernels and tested against plain C++ at every length from 0 to 130 and unaligned addresses (a reversed bit order was shown to be caught). Measured with `bench_codecs` on a Windows desktop PC (AVX2, 768 dimensions): 93 ns per SQ8 L2 distance, 79 ns per SQ8 dot product and 99 ns per binary distance, against 149 ns for floats; without SIMD, binary distances took 4,169 ns, about 9 times slower than floats.

**Q2 (done).** `QuantizationParams::truncate_to` makes an index store and search only the first N dimensions of each vector (re-normalized for cosine); vectors and queries are still given in full, and `get_vector` returns them in full (decision D9), so the full vectors are kept as originals. Each index now distinguishes its input dimension (`dim()`) from its indexed dimension (`indexed_dim()`); one helper, `prepare_truncated`, checks the whole input and prepares its first N values, for every insert and query. Truncation combines with SQ8 (whose codes then cover the first N dimensions) and is saved in `CONF` (format version 3). A truncating index gives results identical to an ordinary index built from the first N values, which test `m01` checks for every metric (and which fails if the wrong dimensions are kept). **Memory:** keeping full vectors (D9) means truncation currently makes searching and building faster but uses more memory, not less; dropping the full vectors would need a per-index option like D6 b. A possible extension is re-ranking at the full dimension ("funnel search"), since the full vectors are kept anyway.

**D6 b, first step (done).** `QuantizationParams::keep_full_vectors = false` drops the copy of each vector as given that cosine indexes and truncating indexes keep. `get_vector`, `get` and `scan` then return the stored vector (normalized for cosine; the first `truncate_to` values when truncating), and `compact` and `search_by_id` rebuild inputs from the stored vectors (padded with zeros when truncating). Search results are identical to keeping the copy, since the stored vectors are the same. Memory: truncating 768 to 256 dimensions now stores about a third as much; cosine indexes store about half. The flag is saved in `CONF`. **Second step (done).** With SQ8 and `keep_full_vectors = false`, training drops the float vectors: `VectorStore` keeps counting slots (so removal, reuse and swap-with-last work unchanged) but stores no values, and reading one throws, a safety net that caught a float read hidden behind a local alias. Every stored vector is read through one helper (floats, or decoded from its code); `distance_between` installs a full-precision marker before comparing two stored vectors; encoding uses the freshly prepared input; duplicates are detected by comparing codes; re-ranking is off. Training again is refused, since it could only use decoded vectors: the model-based test found an index whose 210 vectors had collapsed onto one code after retraining on a tiny sample. Measured: recall@10 0.972 to 0.976 (HNSW; full precision 0.984 to 0.990), and a saved index 3.7 times smaller.

**Q3 (done).** `BinaryCodec`: one bit per dimension, with a learned threshold (mean) and spread (mean absolute deviation) per dimension, asymmetric distances through a per-query table (more accurate than comparing bit strings; a popcount fast path can come with the SIMD work), and exact decode-then-encode, constant dimensions included. Measured recall@10 (3,000 clustered vectors, 64 dimensions): re-ranking is essential and needs many candidates: Flat 0.63 to 0.73 with `rerank_factor` 4, 0.96 to 0.99 with 32; HNSW 0.74 to 0.79 at `ef` 64, 0.98 to 0.99 at `ef` 400; without full vectors 0.34 to 0.39. The model-based test now draws binary configurations too; it showed that binary codes only at 6 to 12 dimensions leave many vectors on the same code, which a graph built from codes cannot separate (documented; the guide recommends keeping full vectors with binary codes).

**Q4 (done).** `PQCodec`: parts of uneven size allowed (`pq_subspaces`, default one per 8 dimensions), 256 centroids per part by k-means (fewer with fewer training vectors), table-lookup distances, deterministic codes (a golden file with PQ codes is reproduced byte for byte on ARM64 too), exact decode then encode (ties toward the lower centroid), and loading that rejects codes naming missing centroids. The number of parts is saved last in `CODC`, keeping earlier offsets (one damaged-file test had to append it to stay a test of its own rule). Measured recall@10 on synthetic clustered data: 0.92 to 0.97 for HNSW at `ef` 200, 0.89 to 0.95 for Flat with `rerank_factor` 16, 0.31 to 0.40 with codes only.

**Q5 (done).** IVF is an option of `FlatIndex` (`IvfParams`) rather than a separate class: an inverted file changes only which rows a scan visits, so it inherits filters, metadata, deadlines, records, bulk changes, thread safety, saving and every codec (IVF-PQ is IVF with `CodecKind::PQ`, decision D10's "full vectors first, then IVF-PQ" in one step). Lists are trained by k-means (all or nothing; repeatable), kept consistent on every slot path in O(1), probed nearest first and beyond `probes` until `k` matches were seen, and saved as `IVFL`. Probing every list gives exactly the exact results (test `i03`); on noisy synthetic data, 16 of 64 lists scanned 25% of the vectors for recall@10 of 0.81. The first version of training could leave the index half-updated if memory ran out (it reused the live list table while building); it now builds everything locally and swaps it in, before any test ran.

**Q6 (done).** In five steps: the Vamana graph (Q6a), the file layout (Q6b), the disk search (Q6c), integration (Q6d: filters, records, batches) and verification (Q6e). Defaults as proposed: built in memory then written; L2 and cosine; plain positional reads, W per round; read-only. Findings: Vamana needed three changes for degenerate data (complete graphs for tiny inputs, identical vectors, and a connectivity repair, since an unreachable vector could never be found); the PQ codes guiding the search must be finer than PQ's general default (one part per 2 dimensions: recall at L 50 went from 0.71 to 0.98 for the same reads); and alpha's benefit did not show at 5,000 vectors (the literature reports it at much larger scales), so it is measured, not asserted. Later extensions: building indexes larger than memory (partition and merge), inner product, and updates in place (FreshDiskANN).

**Extension E1 (done): asynchronous reads.** `DiskOpenOptions::io_threads` reads each round's W nodes concurrently through a small pool shared by all searches, with identical results; a round costs about one read's latency (measured 3.5 times faster with a simulated 3 ms per read and W 4). Threads and positional reads are portable; native asynchronous interfaces (`io_uring`, I/O completion ports) would save the hand-off per read and remain a possible optimization. **Extension E2 (done): building indexes larger than memory.** `build_from_file` partitions a `.fbin` file with k-means, builds a graph per partition within a memory budget, and merges them in one sequential pass, re-pruning with the in-memory PQ codes. Measured on 6,000 vectors from 24 partitions: every vector reachable, recall@10 0.998 at L 100 (as in memory) and 0.950 at L 50 (0.983 in memory). Found by measuring: k-means partitions overran the budget by 1.6 times until full partitions were skipped; 3 replicas overran it again until the partition count was made provably sufficient; and pushing vectors out of full partitions cost recall (0.952) until the automatic count allowed two thirds of slack (1.000).

## Risks

| Risk | Mitigation |
|---|---|
| A long-lived branch drifts from `master` | Merge `master` in regularly; keep every commit green |
| Quantization lowers recall more than expected on some data | Re-ranking at full precision (D6 a); recall measured per codec and metric before each phase is accepted |
| SIMD kernels for codes differ between CPUs | Fixed-answer tests on every CI platform, including the emulated CPUs |
| The disk-based index behaves differently on Windows, macOS and Linux | A small file-access layer with one implementation per platform, tested on every CI job |
| Speed claims cannot be verified in the development environment | Every speed claim is measured on a quiet machine before it is documented |

## Status

| Phase | Status |
|---|---|
| Q0. Foundations | **Done**: k-means, the codec interface and SQ8 in plain C++ (see below) |
| Q1. SQ8 | **Done**, with SIMD kernels (AVX2, AVX-512, NEON) |
| Q2. Matryoshka truncation | **Done** |
| Q3. Binary quantization | **Done** |
| Q4. Product quantization | **Done** |
| Q5. IVF index | **Done**, as an option of `FlatIndex` (IVF-PQ with a PQ codec) |
| Q6. Disk-based index | **Done** (read-only, as planned) |